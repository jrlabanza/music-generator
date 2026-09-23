#!/usr/bin/env python
"""Word timings for lyrics with Whisper (runs in .venv-voice).

    .venv-voice\\Scripts\\python.exe lyrics_sync.py --audio outputs\\x\\audio.flac --output outputs\\x --lyrics-file outputs\\x\\request.json
    .venv-voice\\Scripts\\python.exe lyrics_sync.py --audio uploads\\y\\song.mp3 --output transcriptions\\y

The vocal is separated with Demucs first (cached in <output>/stems) unless --vocals names a stem
or --no-separate says the file is already a vocal. Whisper (openai/whisper-small by default,
MUSICGEN_WHISPER_MODEL to change) transcribes it with word timestamps.

With known lyrics (--lyrics-file: a .txt, or a request.json with a "lyrics" key) the transcript is
aligned to them and karaoke.json, lyrics.lrc and lyrics.srt are written next to words.json.
Without them, the transcription is grouped into lines and sections by its pauses and written to
lyrics.txt. Prints one RESULT line.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

from audio_tools import HF_CACHE, device_name, load_audio, log, resample, separate_stems

DEFAULT_MODEL = os.environ.get("MUSICGEN_WHISPER_MODEL", "openai/whisper-small")


# ── transcription ──────────────────────────────────────────────────────────────
def transcribe_words(vocal_path, model_name, language, device):
    """-> list of {text, start, end} for every word Whisper heard."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
    wave, rate = load_audio(vocal_path)
    mono = resample(wave.mean(0, keepdim=True), rate, 16000)[0].numpy()
    t = time.perf_counter()
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(model_name, cache_dir=str(HF_CACHE))
    model = WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir=str(HF_CACHE), torch_dtype=dtype).to(device)
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor, chunk_length_s=30, stride_length_s=5,
                   device=0 if device == "cuda" else -1, torch_dtype=dtype)
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    out = asr({"raw": mono, "sampling_rate": 16000}, return_timestamps="word", generate_kwargs=generate_kwargs)
    words = []
    for chunk in out.get("chunks", []):
        text = chunk["text"].strip()
        start, end = chunk["timestamp"]
        if not text or start is None:
            continue
        if end is None or end <= start:
            end = start + 0.3
        words.append({"text": text, "start": round(float(start), 3), "end": round(float(end), 3)})
    log(f"whisper {model_name}: {len(words)} words in {time.perf_counter() - t:.0f}s")
    del model, asr
    if device == "cuda":
        torch.cuda.empty_cache()
    return words, out.get("text", "").strip()


# ── alignment with known lyrics ───────────────────────────────────────────────
def norm(token):
    text = unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9']", "", text)


def parse_lyrics(text):
    """-> [{section, text}] for every sung line; [Tags] set the section and are not lines."""
    lines, section = [], ""
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        tag = re.fullmatch(r"\[(.+)\]", line)
        if tag:
            section = tag.group(1).strip()
            continue
        lines.append({"section": section, "text": line})
    return lines


def align(lines, words):
    """Attach start/end times to lyric lines by matching their words to Whisper's words in order."""
    lyric_tokens = []                                        # (line index, normalised token, original)
    for index, line in enumerate(lines):
        for token in re.findall(r"[^\s]+", line["text"]):
            if norm(token):
                lyric_tokens.append((index, norm(token), token))
    heard = [norm(w["text"]) for w in words]
    matcher = difflib.SequenceMatcher(None, [t[1] for t in lyric_tokens], heard, autojunk=False)
    matched = {}
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            matched[a + k] = b + k
    for line in lines:
        line["words"], line["start"], line["end"] = [], None, None
    for position, (index, _, original) in enumerate(lyric_tokens):
        word = {"text": original}
        if position in matched:
            hit = words[matched[position]]
            word["start"], word["end"] = hit["start"], hit["end"]
        lines[index]["words"].append(word)
    for line in lines:
        # Whisper sometimes stamps the first word of a chunk at 0.0 or the last one far too late:
        # take the first/last timed word that sits within 3 s of its neighbour
        starts = [w["start"] for w in line["words"] if "start" in w]
        ends = [w["end"] for w in line["words"] if "end" in w]
        if starts:
            line["start"] = next((s for i, s in enumerate(starts) if i == len(starts) - 1 or starts[i + 1] - s < 3.0), starts[-1])
            line["end"] = next((e for i, e in enumerate(reversed(ends)) if i == len(ends) - 1 or e - list(reversed(ends))[i + 1] < 3.0), ends[0])
            if line["end"] <= line["start"]:
                line["end"] = line["start"] + 0.3
    # fill lines nothing was heard for by spreading them between their timed neighbours
    total = words[-1]["end"] if words else 0.0
    known = [i for i, line in enumerate(lines) if line["start"] is not None]
    if not known:
        return lines, 0.0
    for i, line in enumerate(lines):
        if line["start"] is not None:
            continue
        prev = max((k for k in known if k < i), default=None)
        nxt = min((k for k in known if k > i), default=None)
        lo = lines[prev]["end"] if prev is not None else 0.0
        hi = lines[nxt]["start"] if nxt is not None else max(total, lo + 2.0)
        gap_lines = [j for j in range(i, len(lines)) if lines[j]["start"] is None and (nxt is None or j < nxt)]
        width = max(0.4, (hi - lo) / max(1, len(gap_lines)))
        offset = gap_lines.index(i)
        line["start"], line["end"] = round(lo + offset * width, 3), round(lo + (offset + 1) * width - 0.05, 3)
        line["estimated"] = True
    # keep the timeline monotonic and give every line a sensible end
    for i, line in enumerate(lines):
        if i and line["start"] < lines[i - 1]["end"]:
            line["start"] = lines[i - 1]["end"]
        nxt_start = lines[i + 1]["start"] if i + 1 < len(lines) else None
        if line["end"] is None or line["end"] <= line["start"]:
            line["end"] = (nxt_start - 0.05) if nxt_start else line["start"] + 2.5
        if nxt_start is not None and line["end"] > nxt_start:
            line["end"] = max(line["start"] + 0.2, nxt_start - 0.05)
        line["start"], line["end"] = round(line["start"], 3), round(line["end"], 3)
    ratio = len(matched) / max(1, len(lyric_tokens))
    return lines, ratio


def lrc_text(lines):
    def stamp(t):
        return f"[{int(t // 60):02d}:{t % 60:05.2f}]"
    return "\n".join(f"{stamp(line['start'])}{line['text']}" for line in lines) + "\n"


def srt_text(lines):
    def stamp(t):
        ms = int(round((t - int(t)) * 1000))
        return f"{int(t // 3600):02d}:{int(t // 60) % 60:02d}:{int(t) % 60:02d},{ms:03d}"
    out = []
    for n, line in enumerate(lines, 1):
        out.append(f"{n}\n{stamp(line['start'])} --> {stamp(line['end'])}\n{line['text']}\n")
    return "\n".join(out)


# ── free transcription → lyric-shaped text ────────────────────────────────────
def drop_loops(words, max_repeats=2, longest=12):
    """Whisper can lock into repeating a phrase for the rest of a song; keep at most
    max_repeats consecutive copies of any 2-12 word cycle."""
    keys = [norm(w["text"]) for w in words]
    keep, i = [], 0
    while i < len(words):
        cut = None
        for size in range(2, longest + 1):
            cycle = keys[i:i + size]
            if len(cycle) < size or not any(cycle):
                continue
            repeats = 1
            while keys[i + repeats * size:i + (repeats + 1) * size] == cycle:
                repeats += 1
            if repeats > max_repeats:
                cut = (size, repeats)
                break
        if cut:
            size, repeats = cut
            keep.extend(words[i:i + size * max_repeats])
            i += size * repeats
        else:
            keep.append(words[i])
            i += 1
    return keep


def group_transcript(words, line_gap=0.9, section_gap=2.6, max_words=10):
    words = drop_loops(words)
    sections, line, prev_end = [[]], [], None
    for word in words:
        gap = 0 if prev_end is None else word["start"] - prev_end
        if line and (gap > section_gap):
            sections[-1].append(" ".join(line)); line = []
            sections.append([])
        elif line and (gap > line_gap or len(line) >= max_words):
            sections[-1].append(" ".join(line)); line = []
        line.append(word["text"])
        prev_end = word["end"]
    if line:
        sections[-1].append(" ".join(line))
    sections = [s for s in sections if s]
    text = []
    for k, block in enumerate(sections):
        tag = "Chorus" if (len(sections) > 2 and k % 2 == 1) else "Verse"
        text.append(f"[{tag}]\n" + "\n".join(block))
    return "\n\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, required=True, help="the full mix (or the vocal with --no-separate)")
    parser.add_argument("--vocals", type=Path, help="an already separated vocal stem to use instead of running Demucs")
    parser.add_argument("--no-separate", action="store_true", help="--audio is already a vocal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lyrics-file", type=Path, help="known lyrics: .txt, or a request.json with a lyrics key")
    parser.add_argument("--language", help="Whisper language code such as en or tl (default: auto-detect)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    start = time.perf_counter()
    device = device_name()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.vocals:
        vocal = args.vocals
    elif args.no_separate:
        vocal = args.audio
    else:
        vocal = separate_stems(args.audio, args.output / "stems", device)["vocals"]
    words, transcript = transcribe_words(vocal, args.model, args.language, device)
    (args.output / "words.json").write_text(json.dumps({"model": args.model, "language": args.language, "words": words},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {"words": len(words), "model": args.model, "device": device,
               "seconds": round(words[-1]["end"], 1) if words else 0.0}
    lyrics = None
    if args.lyrics_file:
        raw = args.lyrics_file.read_text(encoding="utf-8")
        lyrics = (json.loads(raw).get("lyrics", "") if args.lyrics_file.suffix == ".json" else raw)
    lines = parse_lyrics(lyrics) if lyrics else []
    if lines:
        if not words:
            raise SystemExit("Whisper heard no words in this vocal, so the lyrics cannot be timed.")
        lines, ratio = align(lines, words)
        (args.output / "karaoke.json").write_text(json.dumps({"lines": lines, "matched_ratio": round(ratio, 3), "model": args.model},
                                                             ensure_ascii=False, indent=1), encoding="utf-8")
        (args.output / "lyrics.lrc").write_text(lrc_text(lines), encoding="utf-8")
        (args.output / "lyrics.srt").write_text(srt_text(lines), encoding="utf-8")
        summary.update({"karaoke": "karaoke.json", "lrc": "lyrics.lrc", "srt": "lyrics.srt", "lines": len(lines),
                        "matched_ratio": round(ratio, 3)})
    else:
        text = group_transcript(words) if words else ""
        (args.output / "lyrics.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")
        summary.update({"lyrics_text": text, "transcript": transcript[:4000]})
    summary["total_seconds"] = round(time.perf_counter() - start, 1)
    print("RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
