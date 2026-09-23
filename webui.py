#!/usr/bin/env python
"""Local web UI for YuE2 on the 8 GB runner.

    YuE\\.venv\\Scripts\\python.exe webui.py              # then open http://127.0.0.1:7860

The model is loaded once into system RAM when the server starts and the GPU
is used by one generation at a time (further requests queue). Every song is
written to outputs/<id>/ with the same artifacts as run_lowvram.py, and the
page lists that folder as a library, so nothing is lost when the server stops.
"""
from __future__ import annotations

import argparse
import atexit
import json
import mimetypes
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from run_lowvram import HERE, MODELS, REPO, VRAM_MODES, gpu_total_gib, make_pipeline, resolve_vram_mode

WEB = HERE / "webui"
OUTPUTS = HERE / "outputs"
TRASH = HERE / "trash"                  # deleted songs are moved here, not destroyed
PLANS = HERE / "plans"                  # scores planned without audio (plan-first workflow)
UPLOADS = HERE / "uploads"              # recordings uploaded for transcription
TRANSCRIPTIONS = HERE / "transcriptions"
EXAMPLES = REPO / "examples"
ABC_TOOLS = REPO / "skills" / "yue2-music" / "scripts" / "abc_tools.py"
LEGACY_VAE = MODELS / "YuE2-Vae-legacy"
SHEETSAGE_PY = HERE / ".venv-sheetsage2" / "Scripts" / "python.exe"
SHEETSAGE_MODEL = MODELS / "SheetSage2"
MAX_UPLOAD_BYTES = 200 * 2**20
VOICES = HERE / "voices"                # reference recordings for voice conversion (personal; never in git)
VOICE_PY = HERE / ".venv-voice" / "Scripts" / "python.exe"
SEEDVC_DIR = HERE / "tools" / "seed-vc"
VOICE_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}


def voice_files():
    """Reference voices on disk: [{name, file, seconds}]."""
    import soundfile as sf
    if not VOICES.is_dir():
        return []
    out = []
    for f in sorted(VOICES.iterdir()):
        if f.is_file() and f.suffix.lower() in VOICE_SUFFIXES:
            try:
                seconds = round(sf.info(str(f)).duration, 1)
            except Exception:
                seconds = None
            out.append({"name": f.stem, "file": f.name, "seconds": seconds})
    return out


def voice_ready():
    return VOICE_PY.is_file() and (SEEDVC_DIR / "inference.py").is_file()


SETTINGS_FILE = HERE / "settings.json"     # notification hooks (personal; never in git)
PRESETS_FILE = HERE / "presets.json"       # the user's own style presets
PRESETS_DEFAULT = WEB / "presets.default.json"
SHARE_SECRET_FILE = HERE / ".share_secret"
WHISPER_MODEL = os.environ.get("MUSICGEN_WHISPER_MODEL", "openai/whisper-small")


def read_settings():
    data = read_json(SETTINGS_FILE) or {}
    return {"discord_webhook": str(data.get("discord_webhook") or ""), "telegram_bot_token": str(data.get("telegram_bot_token") or ""),
            "telegram_chat_id": str(data.get("telegram_chat_id") or ""), "notify_failed": bool(data.get("notify_failed", True))}


def share_secret():
    """A random secret for signed share links, kept across restarts."""
    try:
        secret = SHARE_SECRET_FILE.read_text(encoding="utf-8").strip()
        if len(secret) >= 32:
            return secret.encode("utf-8")
    except OSError:
        pass
    import secrets
    secret = secrets.token_hex(32)
    SHARE_SECRET_FILE.write_text(secret, encoding="utf-8")
    return secret.encode("utf-8")


def make_share_token(song_id, expires):
    import base64
    import hashlib
    import hmac
    payload = base64.urlsafe_b64encode(json.dumps({"s": song_id, "e": int(expires)}).encode("utf-8")).decode("ascii").rstrip("=")
    signature = hmac.new(share_secret(), payload.encode("ascii"), hashlib.sha256).hexdigest()[:32]
    return f"{payload}.{signature}"


def parse_share_token(token):
    """-> song_id, or None when the token is forged or expired."""
    import base64
    import hashlib
    import hmac
    try:
        payload, signature = token.split(".", 1)
        expected = hmac.new(share_secret(), payload.encode("ascii"), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(signature, expected):
            return None
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (ValueError, TypeError):
        return None
    if data.get("e", 0) < time.time() or not re.fullmatch(ID_RE, str(data.get("s", ""))):
        return None
    return data["s"]


def song_meta(directory):
    data = read_json(directory / "meta.json") or {}
    return {"favorite": bool(data.get("favorite")), "rating": int(data.get("rating") or 0),
            "tags": [str(t) for t in (data.get("tags") or [])][:20], "notes": str(data.get("notes") or "")[:4000],
            "owner": data.get("owner"), "origin": data.get("origin")}


def write_meta(directory, **changes):
    data = read_json(directory / "meta.json") or {}
    data.update({k: v for k, v in changes.items() if v is not None})
    (directory / "meta.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def notify(text):
    """Post a line to the configured Discord webhook and/or Telegram chat, in the background."""
    settings = read_settings()
    targets = []
    if settings["discord_webhook"]:
        targets.append((settings["discord_webhook"], {"content": text[:1900]}))
    if settings["telegram_bot_token"] and settings["telegram_chat_id"]:
        targets.append((f"https://api.telegram.org/bot{settings['telegram_bot_token']}/sendMessage",
                        {"chat_id": settings["telegram_chat_id"], "text": text[:4000], "disable_web_page_preview": True}))
    if not targets:
        return False

    def send():
        import urllib.request
        for url, body in targets:
            try:
                request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                                 headers={"Content-Type": "application/json", "User-Agent": "MusicGenStudio/1.0"})
                urllib.request.urlopen(request, timeout=15).read()
            except Exception as exc:
                print(f"  [notify] {type(exc).__name__}: {exc}", file=sys.stderr)
    threading.Thread(target=send, name="notify", daemon=True).start()
    return True


TOKENS_PER_SECOND = 25            # semantic tokens per second of audio (25 Hz latents)
mimetypes.add_type("audio/flac", ".flac")


# ── Pipeline with observable stages ───────────────────────────────────────────
class StageObserver:
    """Mixin placed in front of either pipeline class by make_pipeline(bases=...)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_stage = None
        self.stage_started = None

    @contextmanager
    def _status(self, label, *, total=None, unit=None):
        with super()._status(label, total=total, unit=unit) as stage:
            self.current_stage, self.stage_started = stage, time.monotonic()
            try:
                yield stage
            finally:
                self.current_stage = None

    def preload(self):
        """Load the transformer and VAE into system RAM without touching the GPU."""
        if self._model is None:
            from yue2.modeling_yue2 import YuE2ForCausalLM
            with self._status("Loading model into system RAM"):
                self._model = YuE2ForCausalLM.from_pretrained(
                    self.model_dir, local_files_only=True,
                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        if self._vae is None:
            from yue2.modeling_vae import YuE2VAE
            with self._status("Loading audio decoder"):
                self._vae = YuE2VAE.from_pretrained(self.vae_dir, decoder_only=True, device="cpu",
                                                    local_files_only=True)

    def stage_snapshot(self):
        stage = self.current_stage
        if stage is None:
            return None
        elapsed = time.monotonic() - (self.stage_started or time.monotonic())
        return {"label": stage.label, "completed": stage.completed, "total": stage.total,
                "unit": stage.unit, "elapsed": elapsed,
                "rate": stage.completed / elapsed if elapsed > 0.5 else None}


# ── Jobs ──────────────────────────────────────────────────────────────────────
@dataclass
class Job:
    id: str
    title: str
    request: dict
    kind: str = "song"                 # song | plan | decode | transcribe
    options: dict = field(default_factory=dict)   # sampling overrides, ode_steps, ...
    created: float = field(default_factory=time.time)
    state: str = "queued"              # queued | running | done | failed | cancelled
    error: str | None = None
    started: float | None = None
    finished: float | None = None
    cancel_requested: bool = False
    result: dict | None = None
    owner: str | None = None           # display name of whoever queued it (shared use)

    def public(self):
        return {"id": self.id, "title": self.title, "kind": self.kind, "state": self.state, "error": self.error,
                "created": self.created, "started": self.started, "finished": self.finished, "owner": self.owner,
                "request": {k: v for k, v in self.request.items() if k != "abc"} | ({"abc": True} if self.request.get("abc") else {}),
                "options": self.options, "result": self.result}


class Engine:
    """One worker thread owns the pipeline and all CUDA work."""

    def __init__(self, args):
        self.args = args
        self.pipe = None
        self.vram_mode = resolve_vram_mode(args.vram)
        self.model_state, self.model_error = "loading", None
        self.queue: "queue.Queue[Job]" = queue.Queue()
        self.jobs: dict[str, Job] = {}
        self.current: Job | None = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="yue2-worker", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.pipe = make_pipeline(
                self.args.model, vae=self.args.vae, vram=self.vram_mode, quantization=self.args.quantization,
                gpu_reserve_gib=self.args.gpu_reserve_gib, graph_attention=self.args.graph_attention,
                bases=(StageObserver,))
            self.pipe.preload()
            self.model_state = "ready"
        except Exception as exc:                       # surface to the page instead of dying silently
            self.model_state, self.model_error = "error", f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            return
        work = {"song": self._work_song, "plan": self._work_plan, "decode": self._work_decode,
                "transcribe": self._work_transcribe, "voice": self._work_voice, "stems": self._work_stems,
                "karaoke": self._work_karaoke}
        while True:
            job = self.queue.get()
            if job.cancel_requested:
                job.state, job.finished = "cancelled", time.time()
                continue
            self._run_job(job, work[job.kind])

    def _run_job(self, job, work):
        with self.lock:
            self.current, job.state, job.started = job, "running", time.time()
        try:
            job.result = work(job)
            job.state = "done"
        except InterruptedError:
            job.state = "cancelled"
        except Exception as exc:
            job.state, job.error = "failed", f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            if isinstance(exc, torch.OutOfMemoryError):
                job.error += " — the GPU ran out of memory; try shorter lyrics, or restart the server with --quantization fp8"
        finally:
            job.finished = time.time()
            self._release_gpu()
            sweep_deleted()
            with self.lock:
                self.current = None
            self._after(job)

    def _after(self, job):
        """Bookkeeping once a job has finished: song metadata, follow-up jobs, notifications."""
        try:
            if job.kind == "song" and job.state == "done":
                write_meta(OUTPUTS / job.id, owner=job.owner, origin=job.options.get("origin"))
                voice = job.options.get("then_voice")
                reference = next((v for v in voice_files() if v["name"] == voice), None) if voice else None
                if reference is not None and voice_ready():
                    self.submit(f"{job.title} (in {voice}'s voice)", {"song_id": job.id, "voice": voice, "reference": reference["file"],
                                "semitones": 0, "steps": 30, "auto_f0": True}, kind="voice", owner=job.owner)
            if job.state == "done" or (job.state == "failed" and read_settings()["notify_failed"]):
                notify(job_message(self, job))
        except Exception:
            traceback.print_exc()

    @contextmanager
    def _generation_options(self, options):
        """Apply per-job ODE step overrides to the (single-threaded) pipeline, then restore."""
        import dataclasses
        original = self.pipe.generation_config
        if options.get("ode_steps"):
            self.pipe.generation_config = dataclasses.replace(original, ode_steps=int(options["ode_steps"]))
        try:
            yield
        finally:
            self.pipe.generation_config = original

    def _work_song(self, job):
        out = OUTPUTS / job.id
        with self._generation_options(job.options):
            song = self.pipe(**job.request, abc_sampling=job.options.get("abc_sampling"),
                             semantic_sampling=job.options.get("semantic_sampling"),
                             cancelled=lambda: job.cancel_requested)
        out.mkdir(parents=True, exist_ok=True)
        song.save(out / "audio.flac")               # audio first: a metadata problem must never lose the take
        (out / "title.txt").write_text(job.title, encoding="utf-8")
        song.save_artifacts(out)
        return {"song_id": job.id, "seconds": len(song.audio) / song.sample_rate, "truncated": song.truncated}

    def _work_plan(self, job):
        plan = self.pipe.plan(**job.request, abc_sampling=job.options.get("abc_sampling"),
                              cancelled=lambda: job.cancel_requested)
        out = PLANS / job.id
        plan.save(out)
        (out / "title.txt").write_text(job.title, encoding="utf-8")
        (out / "request.json").write_text(json.dumps(job.request, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"plan_id": job.id, "abc": plan.abc, "tokens": len(plan.abc_ids), "truncated": bool(plan.truncated),
                "seconds": (plan.timing or {}).get("seconds")}

    def _work_decode(self, job):
        import numpy as np
        import soundfile as sf
        song_id, vae_dir = job.request["song_id"], Path(job.request["vae_dir"])
        latents = np.load(OUTPUTS / song_id / "latent.npy")
        audio = self.pipe.decode(latents, vae=str(vae_dir))
        name = job.request.get("output_name", "audio-legacy.flac")
        sf.write(OUTPUTS / song_id / name, audio, 48000, subtype="PCM_24")
        return {"song_id": song_id, "audio": f"/outputs/{song_id}/{name}", "seconds": len(audio) / 48000}

    def _run_tool(self, job, command, label):
        """Run a helper script in another environment; it prints one 'RESULT {json}' line."""
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", cwd=str(HERE))
        lines = []
        for line in process.stdout:
            lines.append(line.rstrip())
            if job.cancel_requested:
                process.terminate()
                raise InterruptedError(f"{label} cancelled")
        process.wait()
        result = next((json.loads(l[7:]) for l in lines if l.startswith("RESULT ")), None)
        if process.returncode or result is None:
            tail = "\n".join(l for l in lines[-15:] if l.strip())
            raise RuntimeError(f"{label} failed:\n" + tail[-1500:])
        return result

    def _work_transcribe(self, job):
        out = TRANSCRIPTIONS / job.id
        command = [str(SHEETSAGE_PY), "-X", "utf8", str(HERE / "sheetsage_transcribe.py"), job.request["path"], "--output", str(out)]
        if not job.request.get("melody_only", True):
            command.append("--full")
        result = self._run_tool(job, command, "SheetSage2")
        if job.request.get("lyrics") and voice_ready():        # words too (Whisper), so a cover needs no typing
            try:
                heard = self._lyrics_sync(job, Path(job.request["path"]), out)
                result["lyrics_text"] = heard.get("lyrics_text", "")
            except InterruptedError:
                raise
            except Exception as exc:
                result.setdefault("warnings", []).append(f"lyrics were not transcribed ({str(exc).splitlines()[-1][:160]})")
        return {"transcription_id": job.id, **result}

    def _lyrics_sync(self, job, audio, out_dir, lyrics_file=None):
        command = [str(VOICE_PY), "-X", "utf8", str(HERE / "lyrics_sync.py"), "--audio", str(audio), "--output", str(out_dir),
                   "--model", WHISPER_MODEL]
        if lyrics_file is not None:
            command += ["--lyrics-file", str(lyrics_file)]
        if job.request.get("language"):
            command += ["--language", str(job.request["language"])]
        return self._run_tool(job, command, "Lyric sync")

    def _work_karaoke(self, job):
        song_id = job.request["song_id"]
        directory = OUTPUTS / song_id
        result = self._lyrics_sync(job, directory / "audio.flac", directory, lyrics_file=directory / "request.json")
        return {**result, "song_id": song_id, "karaoke": f"/outputs/{song_id}/karaoke.json"}

    def _work_stems(self, job):
        song_id = job.request["song_id"]
        command = [str(VOICE_PY), "-X", "utf8", str(HERE / "stems.py"), "--song", str(OUTPUTS / song_id / "audio.flac"),
                   "--output", str(OUTPUTS / song_id)]
        result = self._run_tool(job, command, "Stem separation")
        return {**result, "song_id": song_id, "stems": {k: f"/outputs/{song_id}/{v}" for k, v in result["stems"].items()}}

    def _work_voice(self, job):
        song_id, request = job.request["song_id"], job.request
        directory = OUTPUTS / song_id
        command = [str(VOICE_PY), "-X", "utf8", str(HERE / "voice_convert.py"),
                   "--song", str(directory / "audio.flac"), "--reference", str(VOICES / request["reference"]),
                   "--name", request["voice"], "--output", str(directory),
                   "--semitones", str(request.get("semitones", 0)), "--steps", str(request.get("steps", 30))]
        if request.get("auto_f0"):
            command.append("--auto-f0")
        if request.get("harmonies"):
            command += ["--harmonies", ",".join(str(int(h)) for h in request["harmonies"]),
                        "--harmony-gain-db", str(request.get("harmony_gain_db", -6))]
        if request.get("duet_reference"):
            if not (directory / "karaoke.json").is_file():        # the duet split follows the lyric line timings
                self._lyrics_sync(job, directory / "audio.flac", directory, lyrics_file=directory / "request.json")
            command += ["--duet-reference", str(VOICES / request["duet_reference"]), "--duet-name", request["duet_voice"],
                        "--duet-sections", ",".join(request.get("duet_sections") or ["chorus"]), "--karaoke", str(directory / "karaoke.json")]
        result = self._run_tool(job, command, "Voice conversion")
        return {**result, "song_id": song_id, "audio": f"/outputs/{song_id}/{result['audio']}"}

    def _release_gpu(self):
        """Park the transformer in RAM between jobs so the card is free for other apps."""
        try:
            if self.pipe is not None and self.pipe._model is not None:
                self.pipe._model.to("cpu")
        except Exception:
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # public API -------------------------------------------------------------
    def submit(self, title, request, kind="song", options=None, owner=None):
        if kind in ("song", "plan", "transcribe"):
            job_id = unique_id(title, {"song": OUTPUTS, "plan": PLANS, "transcribe": TRANSCRIPTIONS}[kind])
        else:                                   # decode / voice / stems / karaoke: derived from an existing song
            job_id = f"{kind}-{request['song_id']}-{int(time.time())}"
            while job_id in self.jobs:
                job_id += "x"
        job = Job(id=job_id, title=title, request=request, kind=kind, options=options or {}, owner=owner)
        self.jobs[job.id] = job
        self.queue.put(job)
        return job

    def cancel(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.state in {"queued", "running"}:
            job.cancel_requested = True
        return job

    def status(self):
        gpu = None
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            gpu = {"name": torch.cuda.get_device_name(0), "used_gib": (total - free) / 2**30,
                   "total_gib": total / 2**30, "process_gib": torch.cuda.memory_allocated(0) / 2**30}
        with self.lock:
            job = self.current
        current = job.public() if job else None
        if current and self.pipe is not None:
            current["stage"] = self.pipe.stage_snapshot()
            current["elapsed"] = time.time() - (job.started or time.time())
        queued = [j.public() for j in self.jobs.values() if j.state == "queued"]
        return {"model": {"state": self.model_state, "error": self.model_error,
                          "stage": self.pipe.stage_snapshot() if self.pipe and self.model_state == "loading" else None,
                          "quantization": self.args.quantization, "vram_mode": self.vram_mode},
                "gpu": gpu, "current": current, "queue": queued,
                "share_urls": getattr(self, "share_urls", [])}


# ── Output library ────────────────────────────────────────────────────────────
def slugify(text, fallback="song"):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:40].strip("-")
    return slug or fallback


def unique_id(title, base_dir=OUTPUTS):
    base = f"{datetime.now():%Y%m%d-%H%M%S}-{slugify(title)}"
    candidate, n = base, 2
    while (base_dir / candidate).exists():
        candidate, n = f"{base}-{n}", n + 1
    return candidate


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def song_summary(directory: Path):
    result = read_json(directory / "result.json")
    request = read_json(directory / "request.json")
    if result is None or request is None or not (directory / "audio.flac").is_file() or (directory / ".deleted").exists():
        return None
    title_file = directory / "title.txt"
    title = title_file.read_text(encoding="utf-8").strip() if title_file.is_file() else request.get("id", directory.name)
    meta = song_meta(directory)
    return {"id": directory.name, "title": title, "style": request.get("style", ""),
            "cot": request.get("cot", "full"), "seed": request.get("seed"),
            "seconds": result.get("audio_seconds"), "truncated": result.get("truncated"),
            "created": (directory / "result.json").stat().st_mtime,
            "has_score": (directory / "score.abc").is_file(),
            "elapsed": (result.get("timing") or {}).get("e2e_seconds"),
            "favorite": meta["favorite"], "rating": meta["rating"], "tags": meta["tags"], "owner": meta["owner"],
            "origin": (meta["origin"] or {}).get("song_id") if isinstance(meta["origin"], dict) else None,
            "has_voice": any(directory.glob("audio-voice-*.flac")),
            "has_stems": (directory / "stems" / "accompaniment.flac").is_file(),
            "has_karaoke": (directory / "karaoke.json").is_file()}


def library():
    songs = [s for d in OUTPUTS.iterdir() if d.is_dir() and (s := song_summary(d))] if OUTPUTS.is_dir() else []
    return sorted(songs, key=lambda s: s["created"], reverse=True)


def move_to_trash(directory: Path, attempts=8, delay=0.25):
    """Move a song folder into trash/, retrying briefly: on Windows a folder cannot move
    while any file in it is open, and a browser's aborted download takes a moment to close."""
    TRASH.mkdir(exist_ok=True)
    target = TRASH / directory.name
    if target.exists():
        target = TRASH / f"{directory.name}-{int(time.time())}"
    for attempt in range(attempts):
        try:
            # A plain rename: all-or-nothing. (shutil.move would fall back to copy-then-delete
            # and could leave a half-deleted folder when one file is still open.)
            os.rename(directory, target)
            return target
        except OSError:
            if attempt == attempts - 1:
                return None
            time.sleep(delay)


def sweep_deleted():
    """Finish deferred deletions (folders marked .deleted while a file was still open)."""
    if not OUTPUTS.is_dir():
        return 0
    moved = 0
    for directory in OUTPUTS.iterdir():
        if directory.is_dir() and (directory / ".deleted").exists():
            if move_to_trash(directory, attempts=1):
                moved += 1
    return moved


# ── HTTP ──────────────────────────────────────────────────────────────────────
ID_RE = r"[A-Za-z0-9][A-Za-z0-9_.-]*"


class SamplingOverride(BaseModel):
    temperature: float | None = Field(default=None, ge=0, le=3)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1, le=2000)
    repetition_penalty: float | None = Field(default=None, ge=0.5, le=3)
    max_tokens: int | None = Field(default=None, ge=32, le=9000)

    def overrides(self):
        return {k: v for k, v in self.model_dump().items() if v is not None} or None


class GenerateRequest(BaseModel):
    title: str = ""
    style: str
    lyrics: str
    cot: str = "full"
    seed: int | None = None
    abc: str | None = None
    cfg_scale: float | None = Field(default=None, ge=0, le=20)
    takes: int = Field(default=1, ge=1, le=4)
    abc_sampling: SamplingOverride | None = None
    semantic_sampling: SamplingOverride | None = None
    ode_steps: int | None = Field(default=None, ge=4, le=64)
    then_voice: str | None = None     # sing the finished song in this reference voice automatically
    origin: dict | None = None        # {"song_id", "kind": take|restyle|edit|styles} for the version history


class LyricsRequest(BaseModel):
    task: str = "continue"            # continue | write | rewrite
    lyrics: str = ""
    style: str = ""
    title: str = ""
    section: str | None = None
    language: str | None = None
    text: str | None = None           # section to rewrite


class VoiceRequest(BaseModel):
    voice: str
    semitones: int = Field(default=0, ge=-12, le=12)
    steps: int = Field(default=30, ge=10, le=60)
    auto_f0: bool = False                 # move the melody into the reference singer's range first
    harmonies: list[int] = Field(default_factory=list)      # extra copies at these intervals, e.g. [4, 7]
    harmony_gain_db: float = Field(default=-6.0, ge=-24, le=0)
    duet_voice: str | None = None
    duet_sections: list[str] = Field(default_factory=lambda: ["chorus"])


class ExportRequest(BaseModel):
    format: str = "mp3"                   # mp3 | wav | flac
    source: str = "audio.flac"            # audio.flac or one of the song's audio-*.flac versions
    normalize: bool = True
    lufs: float = Field(default=-14.0, ge=-30, le=-6)
    fade_in: float = Field(default=0.0, ge=0, le=30)
    fade_out: float = Field(default=0.0, ge=0, le=30)
    trim: bool = False


class MetaRequest(BaseModel):
    favorite: bool | None = None
    rating: int | None = Field(default=None, ge=0, le=5)
    tags: list[str] | None = None
    notes: str | None = None


class PresetsRequest(BaseModel):
    presets: list[dict]


class SettingsRequest(BaseModel):
    discord_webhook: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    notify_failed: bool | None = None


class DescribeRequest(BaseModel):
    text: str
    language: str | None = None


class AbcToolRequest(BaseModel):
    action: str
    abc: str
    abc2: str | None = None
    voice: str = "both"
    semitones: int = Field(default=0, ge=-24, le=24)
    bpm: int | None = Field(default=None, ge=20, le=300)
    allow_tempo_change: bool = False


def build_request(body: GenerateRequest):
    """Validate a web request into (title, SongRequest kwargs, generation options) or raise 422."""
    from yue2.protocol import GenerationConfig, resolve_sampling
    style, lyrics = body.style.strip(), body.lyrics.strip()
    if not style or not lyrics:
        raise HTTPException(422, "Style and lyrics are both required.")
    if body.cot not in {"full", "melody", "off"}:
        raise HTTPException(422, "cot must be full, melody or off.")
    abc = body.abc.strip() if body.abc and body.abc.strip() else None
    if abc and body.cot == "off":
        raise HTTPException(422, "A supplied score needs the full or melody plan mode.")
    seed = body.seed if body.seed is not None else random.randrange(2**31)
    if not 0 <= seed < 2**63:
        raise HTTPException(422, "Seed must be in [0, 2**63).")
    title = body.title.strip() or " ".join(lyrics.replace("[", " ").replace("]", " ").split()[:5]) or "song"
    request = {"style": style, "lyrics": lyrics, "cot": body.cot, "seed": seed, "id": slugify(title)}
    if abc:
        request["abc"] = abc
    if body.cfg_scale is not None:
        request["cfg_scale"] = body.cfg_scale
    options, defaults = {}, GenerationConfig()
    for key, default, cap in (("abc_sampling", defaults.abc, 4096), ("semantic_sampling", defaults.semantic, 9000)):
        override = getattr(body, key)
        values = override.overrides() if override else None
        if values:
            if values.get("max_tokens", 0) > cap:
                raise HTTPException(422, f"{key}.max_tokens must be at most {cap}")
            try:
                resolve_sampling(values, default)
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, f"{key}: {exc}")
            options[key] = values
    if body.ode_steps:
        options["ode_steps"] = body.ode_steps
    if body.then_voice:
        options["then_voice"] = slugify(body.then_voice, "voice")
    origin = body.origin or {}
    if isinstance(origin.get("song_id"), str) and re.fullmatch(ID_RE, origin["song_id"]):
        options["origin"] = {"song_id": origin["song_id"], "kind": str(origin.get("kind", ""))[:20]}
    return title, request, options


def build_app(engine: Engine, password: str | None = None):
    app = FastAPI(title="Music Gen Studio")
    OUTPUTS.mkdir(exist_ok=True)
    sweep_deleted()
    if password:
        install_password(app, password)
    @app.get("/outputs/{path:path}")
    def output_file(path: str, request: Request):
        """Serve song files from memory: the file is opened and closed at once, so a
        paused player's suspended download never pins the folder open on Windows."""
        root = OUTPUTS.resolve()
        file = (OUTPUTS / path).resolve()
        if root not in file.parents or not file.is_file() or (file.parent / ".deleted").exists():
            raise HTTPException(404, "Not found")
        data = file.read_bytes()
        size = len(data)
        media = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", request.headers.get("range", "").strip())
        if match and size:
            first, last = match.group(1), match.group(2)
            start = int(first) if first else max(0, size - int(last or 0))
            end = min(int(last), size - 1) if (first and last) else size - 1
            if start > end or start >= size:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            return Response(data[start:end + 1], status_code=206, media_type=media, headers=headers)
        return Response(data, media_type=media, headers=headers)
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB / "index.html").read_text(encoding="utf-8")

    @app.get("/manifest.webmanifest")
    def manifest():
        return FileResponse(WEB / "manifest.webmanifest", media_type="application/manifest+json")

    @app.get("/sw.js")
    def service_worker():
        return FileResponse(WEB / "sw.js", media_type="application/javascript", headers={"Cache-Control": "no-cache"})

    def owner_of(request: Request):
        """Display name the page sends for whoever is using it (shared use); optional."""
        return re.sub(r"\s+", " ", request.headers.get("x-user", "")).strip()[:40] or None

    @app.get("/api/status")
    def status():
        return engine.status()

    @app.get("/api/examples")
    def examples():
        song = read_json(EXAMPLES / "song.json") or {}
        scores = {name: (EXAMPLES / f"{name}.abc").read_text(encoding="utf-8")
                  for name in ("melody", "score", "score-jazz") if (EXAMPLES / f"{name}.abc").is_file()}
        return {"song": song, "scores": scores}

    def ready():
        if engine.model_state == "error":
            raise HTTPException(503, f"Model failed to load: {engine.model_error}")

    @app.post("/api/generate")
    def generate(body: GenerateRequest, http: Request):
        ready()
        title, request, options = build_request(body)
        jobs = []
        for k in range(body.takes):
            req = dict(request)
            if body.takes > 1:
                req["seed"] = request["seed"] + k if body.seed is not None else random.randrange(2**31)
            name = f"{title} (take {k + 1}/{body.takes})" if body.takes > 1 else title
            jobs.append(engine.submit(name, req, options=options, owner=owner_of(http)))
        return {**jobs[0].public(), "jobs": [j.public() for j in jobs]}

    @app.post("/api/plan")
    def plan(body: GenerateRequest, http: Request):
        ready()
        title, request, options = build_request(body)
        if request["cot"] == "off":
            raise HTTPException(422, "Planning a score needs the full or melody mode.")
        if request.get("abc"):
            raise HTTPException(422, "A supplied score already is the plan; render it instead.")
        options = {k: v for k, v in options.items() if k == "abc_sampling"}
        return engine.submit(title, request, kind="plan", options=options, owner=owner_of(http)).public()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        job = engine.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        data = job.public()
        if job.state == "running" and engine.pipe is not None:
            data["stage"] = engine.pipe.stage_snapshot()
        return data

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        try:
            return engine.cancel(job_id).public()
        except KeyError:
            raise HTTPException(404, "Unknown job")

    @app.get("/api/songs")
    def songs():
        sweep_deleted()                  # finish deferred deletions whenever the library is listed
        return library()

    @app.get("/api/songs/{song_id}")
    def song(song_id: str):
        directory = OUTPUTS / song_id
        summary = song_summary(directory) if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", song_id) else None
        if summary is None:
            raise HTTPException(404, "Unknown song")
        request = read_json(directory / "request.json") or {}
        result = read_json(directory / "result.json") or {}
        score = (directory / "score.abc").read_text(encoding="utf-8") if summary["has_score"] else None
        return {**summary, "request": request, "timing": result.get("timing"),
                "score": score, "audio_url": f"/outputs/{song_id}/audio.flac",
                "alt_audio": f"/outputs/{song_id}/audio-legacy.flac" if (directory / "audio-legacy.flac").is_file() else None,
                "has_latent": (directory / "latent.npy").is_file(), "legacy_vae_available": LEGACY_VAE.is_dir(),
                "voice_versions": [{"name": f.stem[len("audio-voice-"):], "url": f"/outputs/{song_id}/{f.name}", "file": f.name,
                                    "modified": f.stat().st_mtime} for f in sorted(directory.glob("audio-voice-*.flac"))],
                "voice_ready": voice_ready(),
                "stems": {name: f"/outputs/{song_id}/stems/{name}.flac" for name in ("vocals", "drums", "bass", "other", "accompaniment")}
                         if summary["has_stems"] and all((directory / "stems" / f"{n}.flac").is_file() for n in ("vocals", "drums", "bass", "other")) else None,
                "karaoke": {"json": f"/outputs/{song_id}/karaoke.json", "lrc": f"/outputs/{song_id}/lyrics.lrc", "srt": f"/outputs/{song_id}/lyrics.srt",
                            "matched_ratio": (read_json(directory / "karaoke.json") or {}).get("matched_ratio")} if summary["has_karaoke"] else None,
                "exports": [{"name": f.name, "url": f"/outputs/{song_id}/export/{f.name}", "bytes": f.stat().st_size}
                            for f in sorted((directory / "export").glob("*")) if f.is_file()] if (directory / "export").is_dir() else [],
                "meta": song_meta(directory), "family": song_family(song_id)}

    @app.get("/api/plans/{plan_id}")
    def plan_detail(plan_id: str):
        directory = PLANS / plan_id
        if not re.fullmatch(ID_RE, plan_id) or not (directory / "score.abc").is_file():
            raise HTTPException(404, "Unknown plan")
        title = directory / "title.txt"
        return {"id": plan_id, "abc": (directory / "score.abc").read_text(encoding="utf-8"),
                "request": read_json(directory / "request.json") or {},
                "title": title.read_text(encoding="utf-8").strip() if title.is_file() else plan_id}

    @app.post("/api/songs/{song_id}/redecode")
    def redecode(song_id: str, http: Request):
        ready()
        directory = OUTPUTS / song_id
        if not re.fullmatch(ID_RE, song_id) or not (directory / "latent.npy").is_file():
            raise HTTPException(404, "No saved latents for that song")
        if not LEGACY_VAE.is_dir():
            raise HTTPException(501, "The legacy decoder is not downloaded (run download_models.py --legacy-vae).")
        title = directory / "title.txt"
        name = title.read_text(encoding="utf-8").strip() if title.is_file() else song_id
        return engine.submit(f"{name} (legacy decoder)", {"song_id": song_id, "vae_dir": str(LEGACY_VAE),
                             "output_name": "audio-legacy.flac"}, kind="decode", owner=owner_of(http)).public()

    @app.post("/api/abc/tool")
    def abc_tool(body: AbcToolRequest):
        return run_abc_tool(body)

    @app.post("/api/transcribe")
    async def transcribe(http: Request, file: UploadFile = File(...), melody_only: bool = Form(True), title: str = Form(""),
                         lyrics: bool = Form(False), language: str = Form("")):
        if not SHEETSAGE_PY.is_file() or not (SHEETSAGE_MODEL / "config.json").is_file():
            raise HTTPException(501, "SheetSage2 is not set up on this PC (see README: Cover a recording).")
        ready()
        original = Path(file.filename or "recording")
        suffix = original.suffix.lower()
        if suffix not in {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".aiff", ".aif", ".wma", ".webm"}:
            raise HTTPException(422, f"“{original.name}” is not an audio file this app can take — upload wav, flac, mp3 or ogg "
                                     "(m4a/aac/opus/webm need FFmpeg on the PC).")
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", original.stem).strip("-.")[:70] or "recording"   # keep the extension when shortening
        name = stem + suffix
        data = await file.read()
        if not data:
            raise HTTPException(422, "The file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Keep uploads under {MAX_UPLOAD_BYTES // 2**20} MB.")
        stem = title.strip() or Path(name).stem
        folder = UPLOADS / unique_id(stem, UPLOADS)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
        return engine.submit(f"Transcribe {stem}", {"path": str(folder / name), "melody_only": melody_only, "filename": name,
                             "lyrics": bool(lyrics), "language": re.sub(r"[^a-z]", "", language.lower())[:5] or None},
                             kind="transcribe", owner=owner_of(http)).public()

    @app.post("/api/lyrics")
    def lyrics(body: LyricsRequest):
        running, has_model, _ = ollama_status()
        if not running:
            raise HTTPException(501, "The local lyric model is not running: install Ollama (ollama.com) and start it, then try again.")
        if not has_model:
            raise HTTPException(501, f"The lyric model is not downloaded yet: run `ollama pull {LYRICS_MODEL}`.")
        if body.task == "continue" and not body.lyrics.strip():
            raise HTTPException(422, "There are no lyrics to continue from — write a first section, or use Write from title.")
        if body.task == "write" and not (body.title.strip() or body.style.strip()):
            raise HTTPException(422, "Give at least a title or a style to write lyrics from.")
        with engine.lock:
            gpu_free = engine.current is None
        started = time.time()
        text = write_lyrics(body.task, body.lyrics, body.style, body.title, body.section, body.language, body.text, gpu_free)
        return {"text": text, "model": LYRICS_MODEL, "device": "gpu" if gpu_free else "cpu", "seconds": round(time.time() - started, 1)}

    # ── voices (sing a song in your own voice) ─────────────────────────────
    @app.get("/api/voices")
    def voices():
        return {"voices": voice_files(), "ready": voice_ready()}

    @app.post("/api/voices")
    async def add_voice(file: UploadFile = File(...), name: str = Form("")):
        import soundfile as sf
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in VOICE_SUFFIXES:
            raise HTTPException(422, "Upload a wav, flac, mp3 or ogg recording.")
        data = await file.read()
        if not data:
            raise HTTPException(422, "The file is empty.")
        if len(data) > 50 * 2**20:
            raise HTTPException(413, "Keep voice clips under 50 MB (a 30-60 s recording is ideal).")
        voice = slugify(name.strip() or Path(file.filename).stem, "voice")
        VOICES.mkdir(parents=True, exist_ok=True)
        target = VOICES / f"{voice}{suffix}"
        for old in VOICES.glob(f"{voice}.*"):          # replacing a voice keeps one file per name
            old.unlink()
        target.write_bytes(data)
        try:
            seconds = sf.info(str(target)).duration
        except Exception as error:
            target.unlink(missing_ok=True)
            raise HTTPException(422, f"Could not read that audio file: {error}")
        if seconds < 5:
            target.unlink(missing_ok=True)
            raise HTTPException(422, "That clip is too short — record at least 20 s of singing or speaking.")
        if seconds > 180:
            target.unlink(missing_ok=True)
            raise HTTPException(422, "Keep the clip under 3 minutes; 30-60 s of clean audio works best.")
        return {"voice": {"name": voice, "file": target.name, "seconds": round(seconds, 1)}, "voices": voice_files()}

    @app.delete("/api/voices/{name}")
    def delete_voice(name: str):
        matches = [v for v in voice_files() if v["name"] == name]
        if not matches:
            raise HTTPException(404, "Unknown voice")
        for v in matches:
            (VOICES / v["file"]).unlink(missing_ok=True)
        return {"deleted": name, "voices": voice_files()}

    def song_dir(song_id):
        directory = OUTPUTS / song_id
        if not re.fullmatch(ID_RE, song_id) or not (directory / "audio.flac").is_file() or (directory / ".deleted").exists():
            raise HTTPException(404, "Unknown song")
        return directory

    def song_title(directory):
        title = directory / "title.txt"
        return title.read_text(encoding="utf-8").strip() if title.is_file() else directory.name

    def need_voice_env():
        if not voice_ready():
            raise HTTPException(501, "The voice environment is not set up on this PC (see README: Sing it in your voice).")

    @app.post("/api/songs/{song_id}/voice")
    def sing_in_voice(song_id: str, body: VoiceRequest, http: Request):
        ready()
        need_voice_env()
        directory = song_dir(song_id)
        voices = {v["name"]: v for v in voice_files()}
        if body.voice not in voices:
            raise HTTPException(404, "Unknown voice — upload a reference clip first.")
        harmonies = sorted({int(h) for h in body.harmonies if -12 <= int(h) <= 12 and int(h) != 0})[:3]
        request = {"song_id": song_id, "voice": body.voice, "reference": voices[body.voice]["file"], "semitones": body.semitones,
                   "steps": body.steps, "auto_f0": body.auto_f0, "harmonies": harmonies, "harmony_gain_db": body.harmony_gain_db}
        extras = []
        if harmonies:
            extras.append("harmonies")
        if body.duet_voice:
            if body.duet_voice not in voices:
                raise HTTPException(404, "Unknown duet voice.")
            if body.duet_voice == body.voice:
                raise HTTPException(422, "Pick a different voice for the duet.")
            sections = [re.sub(r"[^A-Za-z -]", "", s)[:20] for s in body.duet_sections if s.strip()] or ["chorus"]
            request.update({"duet_voice": body.duet_voice, "duet_reference": voices[body.duet_voice]["file"], "duet_sections": sections})
            extras.append(f"duet with {body.duet_voice}")
        title = f"{song_title(directory)} (in {body.voice}'s voice{', ' + ', '.join(extras) if extras else ''})"
        return engine.submit(title, request, kind="voice", owner=owner_of(http)).public()

    @app.post("/api/songs/{song_id}/stems")
    def split_stems(song_id: str, http: Request):
        ready()
        need_voice_env()
        directory = song_dir(song_id)
        return engine.submit(f"{song_title(directory)} (stems)", {"song_id": song_id}, kind="stems", owner=owner_of(http)).public()

    @app.post("/api/songs/{song_id}/karaoke")
    def karaoke(song_id: str, http: Request, language: str = ""):
        ready()
        need_voice_env()
        directory = song_dir(song_id)
        lyrics = (read_json(directory / "request.json") or {}).get("lyrics", "")
        if not re.sub(r"\[[^\]]*\]", "", lyrics).strip():
            raise HTTPException(422, "This song has no sung lyrics to time (only section tags).")
        request = {"song_id": song_id, "language": re.sub(r"[^a-z]", "", language.lower())[:5] or None}
        return engine.submit(f"{song_title(directory)} (karaoke sync)", request, kind="karaoke", owner=owner_of(http)).public()

    @app.post("/api/songs/{song_id}/export")
    def export(song_id: str, body: ExportRequest):
        need_voice_env()
        directory = song_dir(song_id)
        if body.format not in ("mp3", "wav", "flac"):
            raise HTTPException(422, "format must be mp3, wav or flac")
        source = Path(body.source).name
        if not re.fullmatch(r"audio(-[A-Za-z0-9_.-]+)?\.flac", source) or not (directory / source).is_file():
            raise HTTPException(404, "Unknown audio version")
        variant = source[len("audio-"):-len(".flac")] if source != "audio.flac" else ""
        name = slugify(song_title(directory)) + (f"-{variant}" if variant else "") + ("-master" if body.normalize else "")
        command = [str(VOICE_PY), "-X", "utf8", str(HERE / "export_audio.py"), "--song", str(directory / source), "--output", str(directory),
                   "--format", body.format, "--name", name, "--title", song_title(directory), "--fade-in", str(body.fade_in),
                   "--fade-out", str(body.fade_out), "--comment", (read_json(directory / "request.json") or {}).get("style", "")[:200]]
        if body.normalize:
            command += ["--lufs", str(body.lufs)]
        if body.trim:
            command.append("--trim")
        run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(HERE), timeout=600)
        result = next((json.loads(l[7:]) for l in run.stdout.splitlines() if l.startswith("RESULT ")), None)
        if run.returncode or result is None:
            raise HTTPException(500, "Export failed: " + (run.stderr or run.stdout).strip()[-600:])
        return {**result, "url": f"/outputs/{song_id}/{result['file']}", "download": f"{name}.{body.format}"}

    @app.patch("/api/songs/{song_id}/meta")
    def update_meta(song_id: str, body: MetaRequest):
        directory = song_dir(song_id)
        changes = {}
        if body.favorite is not None:
            changes["favorite"] = bool(body.favorite)
        if body.rating is not None:
            changes["rating"] = int(body.rating)
        if body.tags is not None:
            changes["tags"] = sorted({re.sub(r"\s+", " ", t).strip()[:30] for t in body.tags if t.strip()})[:20]
        if body.notes is not None:
            changes["notes"] = body.notes[:4000]
        write_meta(directory, **changes)
        return song_meta(directory)

    @app.get("/api/songs/{song_id}/share")
    def share_link(song_id: str, http: Request, days: int = 7):
        directory = song_dir(song_id)
        days = max(1, min(int(days), 365))
        token = make_share_token(song_id, time.time() + days * 86400)
        public = next((u for u in getattr(engine, "share_urls", []) if "trycloudflare" in u or u.startswith("https://")), None)
        base = (public or str(http.base_url)).rstrip("/")
        return {"url": f"{base}/s/{token}", "expires": int(time.time() + days * 86400), "days": days, "title": song_title(directory)}

    @app.get("/s/{token}", response_class=HTMLResponse)
    def shared_page(token: str):
        song_id = parse_share_token(token)
        directory = OUTPUTS / song_id if song_id else None
        if directory is None or not (directory / "audio.flac").is_file() or (directory / ".deleted").exists():
            return HTMLResponse(SHARE_PAGE.replace("{title}", "Link expired").replace("{body}", "<p>This share link is no longer valid.</p>"), status_code=404)
        request = read_json(directory / "request.json") or {}
        from html import escape
        title = escape(song_title(directory))
        versions = [("Original", "audio.flac")] + [(f"In {f.stem[len('audio-voice-'):]}'s voice", f.name) for f in sorted(directory.glob("audio-voice-*.flac"))]
        players = "".join(f'<div class="v"><div class="sub">{escape(label)}</div><audio controls preload="metadata" src="/s/{token}/audio?v={escape(file)}"></audio></div>'
                          for label, file in versions)
        body = (f'<p class="style">{escape(request.get("style", ""))}</p>{players}'
                f'<p><a class="btn" href="/s/{token}/audio?v=audio.flac&download=1">Download FLAC</a></p>')
        return SHARE_PAGE.replace("{title}", title).replace("{body}", body)

    @app.get("/s/{token}/audio")
    def shared_audio(token: str, request: Request, v: str = "audio.flac", download: int = 0):
        song_id = parse_share_token(token)
        name = Path(v).name
        if song_id is None or not re.fullmatch(r"audio(-[A-Za-z0-9_.-]+)?\.flac", name):
            raise HTTPException(404, "Not found")
        response = output_file(f"{song_id}/{name}", request)
        if download:
            response.headers["Content-Disposition"] = f'attachment; filename="{song_id}.flac"'
        return response

    # ── presets, settings, describe ────────────────────────────────────────
    @app.get("/api/presets")
    def presets():
        return {"builtin": read_json(PRESETS_DEFAULT) or [], "user": read_json(PRESETS_FILE) or []}

    @app.put("/api/presets")
    def save_presets(body: PresetsRequest):
        clean = []
        for item in body.presets[:200]:
            name, style = str(item.get("name", "")).strip()[:60], str(item.get("style", "")).strip()[:500]
            if name and style:
                clean.append({"name": name, "style": style})
        PRESETS_FILE.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"user": clean}

    def masked(value):
        return ("•" * 8 + value[-4:]) if len(value) > 8 else ("•" * len(value) if value else "")

    @app.get("/api/settings")
    def get_settings():
        settings = read_settings()
        return {**settings, "discord_webhook": masked(settings["discord_webhook"]), "telegram_bot_token": masked(settings["telegram_bot_token"]),
                "discord_set": bool(settings["discord_webhook"]), "telegram_set": bool(settings["telegram_bot_token"] and settings["telegram_chat_id"])}

    @app.put("/api/settings")
    def put_settings(body: SettingsRequest):
        settings = read_settings()
        for key in ("discord_webhook", "telegram_bot_token", "telegram_chat_id"):
            value = getattr(body, key)
            if value is None or value.startswith("•"):         # untouched masked value keeps the stored secret
                continue
            settings[key] = value.strip()
        if body.discord_webhook and not body.discord_webhook.startswith("•") and not re.match(r"https://(discord\.com|discordapp\.com)/api/webhooks/", body.discord_webhook.strip()):
            raise HTTPException(422, "That does not look like a Discord webhook URL (https://discord.com/api/webhooks/...).")
        if body.notify_failed is not None:
            settings["notify_failed"] = bool(body.notify_failed)
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        return get_settings()

    @app.post("/api/settings/test")
    def test_settings():
        if not notify("🎵 Music Gen Studio: notifications are working."):
            raise HTTPException(422, "No notification target is configured yet.")
        return {"sent": True}

    @app.post("/api/describe")
    def describe(body: DescribeRequest):
        running, has_model, _ = ollama_status()
        if not running:
            raise HTTPException(501, "The local lyric model is not running: install Ollama (ollama.com) and start it, then try again.")
        if not has_model:
            raise HTTPException(501, f"The lyric model is not downloaded yet: run `ollama pull {LYRICS_MODEL}`.")
        if len(body.text.strip()) < 3:
            raise HTTPException(422, "Describe the song in a sentence or two first.")
        with engine.lock:
            gpu_free = engine.current is None
        started = time.time()
        draft = describe_song(body.text.strip()[:2000], body.language, gpu_free)
        return {**draft, "model": LYRICS_MODEL, "device": "gpu" if gpu_free else "cpu", "seconds": round(time.time() - started, 1)}

    @app.get("/api/doctor")
    def doctor():
        import platform
        gpu = None
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            gpu = {"name": torch.cuda.get_device_name(0), "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                   "total_gib": round(total / 2**30, 2), "free_gib": round(free / 2**30, 2)}

        def folder(path):
            files = [f for f in path.rglob("*") if f.is_file()] if path.is_dir() else []
            return {"path": str(path), "present": bool(files), "size_gb": round(sum(f.stat().st_size for f in files) / 2**30, 2)}
        import importlib.metadata as meta
        versions = {}
        for pkg in ("yue2-infer", "torch", "transformers", "fastapi", "huggingface-hub"):
            try:
                versions[pkg] = meta.version(pkg)
            except meta.PackageNotFoundError:
                versions[pkg] = None
        return {"python": platform.python_version(), "platform": platform.platform(), "versions": versions,
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                "gpu": gpu, "vram_mode": engine.vram_mode, "quantization": engine.args.quantization,
                "graph_attention": engine.args.graph_attention, "model_state": engine.model_state, "model_error": engine.model_error,
                "weights": getattr(engine.pipe, "weights", None),
                "models": {name: folder(path) for name, path in (("YuE2-3B", MODELS / "YuE2-3B"), ("YuE2-Vae", MODELS / "YuE2-Vae"),
                           ("YuE2-Vae-legacy", LEGACY_VAE), ("SheetSage2", SHEETSAGE_MODEL), ("MERT-v2-FullSong", MODELS / "MERT-v2-FullSong"))},
                "lyrics_model": dict(zip(("running", "downloaded", "models"), ollama_status()), name=LYRICS_MODEL),
                "voice_conversion": {"env": VOICE_PY.is_file(), "seed_vc": (SEEDVC_DIR / "inference.py").is_file(), "voices": len(voice_files())},
                "whisper_model": WHISPER_MODEL,
                "notifications": {"discord": bool(read_settings()["discord_webhook"]),
                                  "telegram": bool(read_settings()["telegram_bot_token"] and read_settings()["telegram_chat_id"])},
                "sheetsage2_env": SHEETSAGE_PY.is_file(), "ffmpeg": bool(shutil.which("ffmpeg")) or (HERE / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe").is_file(),
                "cloudflared": str(find_cloudflared() or ""), "share_urls": getattr(engine, "share_urls", []),
                "disk_free_gb": round(shutil.disk_usage(str(HERE)).free / 2**30, 1),
                "songs": len(library()), "plans": len([d for d in PLANS.iterdir() if d.is_dir()]) if PLANS.is_dir() else 0,
                "trash": len([d for d in TRASH.iterdir() if d.is_dir()]) if TRASH.is_dir() else 0}

    @app.delete("/api/songs/{song_id}")
    def delete_song(song_id: str):
        directory = OUTPUTS / song_id
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", song_id) or not directory.is_dir():
            raise HTTPException(404, "Unknown song")
        job = engine.jobs.get(song_id)
        if job is not None and job.state == "running":
            raise HTTPException(409, "That song is still generating; cancel it instead.")
        target = move_to_trash(directory)
        if target is not None:
            return {"deleted": song_id, "moved_to": str(target), "deferred": False}
        # Something still holds a file open (a suspended download, a media player). Hide the
        # song now and finish the move when the handle is released.
        (directory / ".deleted").write_text(str(time.time()), encoding="utf-8")
        return {"deleted": song_id, "moved_to": None, "deferred": True}

    @app.get("/api/qr.svg")
    def qr(text: str):
        if len(text) > 512 or not re.match(r"https?://", text):
            raise HTTPException(422, "text must be an http(s) URL")
        import qrcode
        import qrcode.image.svg
        image = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, box_size=12, border=2)
        return Response(image.to_string(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    @app.get("/api/songs/{song_id}/audio.wav")
    def wav(song_id: str):
        import io
        import soundfile as sf
        path = OUTPUTS / song_id / "audio.flac"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", song_id) or not path.is_file():
            raise HTTPException(404, "Unknown song")
        audio, rate = sf.read(path)
        buffer = io.BytesIO()
        sf.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
        return Response(buffer.getvalue(), media_type="audio/wav",
                        headers={"Content-Disposition": f'attachment; filename="{song_id}.wav"'})

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on (see --share)")
    parser.add_argument("--share", action="store_true",
                        help="listen on all interfaces so people on your network can use the page; "
                             "Windows Firewall must allow TCP port 7860 (see README)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--password", default=os.environ.get("MUSICGEN_PASSWORD") or None,
                        help="require this password (HTTP Basic auth, any username) for every page and download; "
                             "also read from the MUSICGEN_PASSWORD environment variable. Use it whenever the app "
                             "is reachable beyond your own machine")
    parser.add_argument("--model", default=str(MODELS / "YuE2-3B"))
    parser.add_argument("--vae", default=str(MODELS / "YuE2-Vae"))
    parser.add_argument("--vram", choices=VRAM_MODES, default="auto",
                        help="low: swap model halves through system RAM (8 GB cards); "
                             "normal: whole model on the GPU (16 GB+); auto: pick by detected VRAM")
    parser.add_argument("--quantization", choices=("none", "fp8"), default="none")
    parser.add_argument("--gpu-reserve-gib", type=float, default=2.0)
    parser.add_argument("--graph-attention", choices=("cudnn", "sdpa"), default="cudnn",
                        help="cudnn is fast; sdpa is seed-reproducible but ~2.7x slower")
    parser.add_argument("--password-file", type=Path, help="read the password from this file (first line)")
    parser.add_argument("--tunnel", action="store_true",
                        help="publish the app on the internet through a Cloudflare quick tunnel (needs cloudflared and a password); "
                             "the public https address is printed and shown in the share pill with a QR code")
    parser.add_argument("--cloudflared", type=Path, help="path to cloudflared.exe (default: PATH or the usual install folders)")
    parser.add_argument("--keep-awake", dest="keep_awake", action="store_true", default=None,
                        help="stop Windows from sleeping while the app runs (default: on with --share/--tunnel)")
    parser.add_argument("--no-keep-awake", dest="keep_awake", action="store_false")
    parser.add_argument("--open", action="store_true", help="open the page in your browser once the server is up")
    args = parser.parse_args()
    for path in (args.model, args.vae):
        if not Path(path).is_dir():
            sys.exit(f"Model directory not found: {path} -- run download_models.py first (about 7.3 GB).")
    import uvicorn
    if args.password_file:
        args.password = args.password_file.read_text(encoding="utf-8").splitlines()[0].strip() or None
    if args.tunnel and not args.password:
        sys.exit("--tunnel publishes the app on the internet: set --password (or --password-file / MUSICGEN_PASSWORD) first.")
    if args.share:
        args.host = "0.0.0.0"
    engine = Engine(args)
    if args.tunnel:
        exe = find_cloudflared(args.cloudflared)
        if exe is None:
            sys.exit("cloudflared.exe not found: install it from https://github.com/cloudflare/cloudflared/releases "
                     "or pass --cloudflared PATH")
        Tunnel(exe, args.port, engine)
    if args.keep_awake or (args.keep_awake is None and (args.share or args.tunnel)):
        print("  Keeping the PC awake while the app runs" if keep_awake() else "  (could not request keep-awake)", file=sys.stderr)
    engine.share_urls = [f"http://{ip}:{args.port}" for ip in lan_addresses()] if args.host == "0.0.0.0" else []
    browse_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{browse_host}:{args.port}"
    print(f"\n  Music Gen Studio -> {url}\n  GPU {gpu_total_gib():.1f} GiB -> {engine.vram_mode} VRAM mode"
          f"{' (forced)' if args.vram != 'auto' else ''}", file=sys.stderr)
    if engine.share_urls:
        print("  Share on your network -> " + "  or  ".join(engine.share_urls)
              + f"\n  (others need Windows Firewall to allow TCP port {args.port}; see README)", file=sys.stderr)
    print(file=sys.stderr)
    if args.open:
        threading.Thread(target=open_when_ready, args=(browse_host, args.port, url), daemon=True).start()
    if args.password:
        print("  Password protection is on (sign-in page at /login)\n", file=sys.stderr)
    elif args.host == "0.0.0.0":
        print("  No password set: anyone who can reach this address can use it (--password to require one)\n", file=sys.stderr)
    uvicorn.run(build_app(engine, args.password), host=args.host, port=args.port, log_level="warning")


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LYRICS_MODEL = os.environ.get("MUSICGEN_LYRICS_MODEL", "qwen2.5:7b")
LYRICIST_SYSTEM = (
    "You are a professional lyricist who writes singable lyrics with clear rhyme, steady line lengths and a story "
    "that moves forward. Always keep the language of the existing lyrics exactly (Tagalog stays Tagalog, English "
    "stays English, a mix stays a mix) and keep the same voice and tone. Output only lyrics: each section starts "
    "with its tag in square brackets on its own line, such as [Verse], [Chorus], [Bridge] or [Outro], followed by "
    "the lines. No title, no explanations, no quotation marks, no markdown."
)


def ollama_status():
    """(running, has_model, models) for the local lyric model."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as response:
            models = [m.get("name", "") for m in json.loads(response.read()).get("models", [])]
    except Exception:
        return False, False, []
    wanted = LYRICS_MODEL if ":" in LYRICS_MODEL else LYRICS_MODEL + ":latest"
    return True, any(m == wanted or m == LYRICS_MODEL for m in models), models


def clean_lyrics(text, default_tag):
    lines = [l.rstrip() for l in text.replace("\r", "").split("\n")]
    lines = [l for l in lines if not l.strip().startswith("```")]
    while lines and not lines[0].strip():
        lines.pop(0)
    # drop chatter before the first section tag ("Here is the next verse:")
    first_tag = next((i for i, l in enumerate(lines) if l.strip().startswith("[") and l.strip().endswith("]")), None)
    if first_tag is not None:
        lines = lines[first_tag:]
    elif lines and lines[0].strip().endswith(":") and len(lines) > 1:
        lines = lines[1:]
    if not lines or not lines[0].strip().startswith("["):
        lines.insert(0, f"[{default_tag}]")
    out, blank = [], 0
    for line in lines:
        blank = blank + 1 if not line.strip() else 0
        if blank <= 1:
            out.append(line.strip("\"“” "))
    return "\n".join(out).strip()[:4000]


def write_lyrics(task, lyrics="", style="", title="", section=None, language=None, text=None, gpu_free=True):
    """Ask the local Ollama model for lyrics; returns cleaned text."""
    import urllib.request
    context = [f"Title: {title}" if title else "", f"Style: {style}" if style else "", f"Language: {language}" if language else ""]
    context = "\n".join(c for c in context if c)
    if task == "continue":
        want = f"the next section: {section}" if section else "the next section (choose what should come next: another verse, the chorus, a pre-chorus, a bridge or an outro)"
        user = (f"{context}\n\nLyrics so far:\n{lyrics.strip()}\n\nWrite ONLY {want}, 4 to 8 lines, matching the rhyme scheme, "
                "syllable feel and story of the lyrics so far. Start with its section tag.")
        default_tag = section or "Verse"
    elif task == "write":
        user = (f"{context}\n\nWrite complete song lyrics with this structure: [Verse], [Chorus], [Verse], [Chorus], [Bridge], [Chorus]. "
                "Four lines per section; the chorus repeats with the same words each time. Infer the language from the title and "
                "style when none is given (default English).")
        default_tag = "Verse"
    elif task == "rewrite":
        user = (f"{context}\n\nRewrite this section, keeping its meaning, language and line count but improving the rhyme and flow:\n"
                f"{(text or '').strip()}\n\nOutput only the rewritten section with its tag.")
        default_tag = section or "Verse"
    else:
        raise HTTPException(422, "task must be continue, write or rewrite")
    body = {"model": LYRICS_MODEL, "stream": False, "keep_alive": 0,
            "messages": [{"role": "system", "content": LYRICIST_SYSTEM}, {"role": "user", "content": user}],
            "options": {"temperature": 0.85, "top_p": 0.9, "repeat_penalty": 1.1, "num_predict": 450,
                        "num_gpu": 999 if gpu_free else 0}}
    request = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            reply = json.loads(response.read())
    except Exception as exc:
        raise HTTPException(502, f"The lyric model did not answer: {exc}")
    content = (reply.get("message") or {}).get("content", "")
    if not content.strip():
        raise HTTPException(502, "The lyric model returned nothing")
    return clean_lyrics(content, default_tag)


DESCRIBE_SYSTEM = (
    "You turn a short description of a wanted song into a complete brief for a music generator. Reply with JSON only, "
    'with exactly these keys: "title" (2-5 words), "style" (ONE line in this exact shape: language, genre, vocal type and '
    "gender, mood, 2-4 key instruments, tempo in BPM — e.g. \"Tagalog, OPM acoustic ballad, warm male vocal, tender, acoustic "
    'guitar, piano, strings, 72 BPM\"; for an instrumental write "instrumental" instead of a vocal and add "no vocals"), '
    '"language" (the lyric language), and "lyrics" (complete singable lyrics with section tags in square brackets on their '
    "own lines: [Verse], [Chorus], [Verse], [Chorus], [Bridge], [Chorus]; four lines per section, the chorus repeated with the "
    "same words; for an instrumental, just the tags with 'instrumental, no vocals' after each). Write lyrics in the language "
    "the description asks for, or the description's own language. No markdown, no explanations."
)


def describe_song(text, language=None, gpu_free=True):
    """One-shot song brief from a plain description via the local Ollama model -> {title, style, lyrics, language}."""
    import urllib.request
    user = f"Description: {text}" + (f"\nLyric language: {language}" if language else "")
    body = {"model": LYRICS_MODEL, "stream": False, "keep_alive": 0, "format": "json",
            "messages": [{"role": "system", "content": DESCRIBE_SYSTEM}, {"role": "user", "content": user}],
            "options": {"temperature": 0.8, "top_p": 0.9, "repeat_penalty": 1.1, "num_predict": 900,
                        "num_gpu": 999 if gpu_free else 0}}
    request = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            reply = json.loads(response.read())
        draft = json.loads((reply.get("message") or {}).get("content", "") or "{}")
    except Exception as exc:
        raise HTTPException(502, f"The lyric model did not answer usefully: {exc}")
    if not isinstance(draft, dict):
        raise HTTPException(502, "The lyric model did not return a song brief")
    lyrics = draft.get("lyrics", "")
    if isinstance(lyrics, list):
        lyrics = "\n".join(str(l) for l in lyrics)
    style = re.sub(r"\s+", " ", str(draft.get("style", ""))).strip(" .")
    if not style or not str(lyrics).strip():
        raise HTTPException(502, "The lyric model left the style or lyrics empty — try describing the song differently.")
    return {"title": str(draft.get("title", "")).strip()[:80], "style": style[:400],
            "lyrics": clean_lyrics(str(lyrics), "Verse"), "language": str(draft.get("language", "") or language or "")[:40]}


def job_message(engine, job):
    """One line for Discord/Telegram when a job finishes."""
    kinds = {"song": "Song", "plan": "Score plan", "decode": "Re-decode", "transcribe": "Transcription", "voice": "Voice version",
             "stems": "Stems", "karaoke": "Karaoke sync"}
    what = kinds.get(job.kind, job.kind)
    if job.state == "failed":
        return f"❌ {what} failed: {job.title}\n{(job.error or '')[:300]}"
    who = f" for {job.owner}" if job.owner else ""
    took = f" in {int(job.finished - job.started)}s" if job.started and job.finished else ""
    seconds = (job.result or {}).get("seconds")
    length = f" · {int(seconds // 60)}:{int(seconds % 60):02d}" if seconds else ""
    song_id = job.id if job.kind == "song" else (job.request or {}).get("song_id")
    urls = getattr(engine, "share_urls", []) or []
    base = next((u for u in urls if "trycloudflare" in u or u.startswith("https://")), None) or (urls[0] if urls else None)
    link = f"\n{base}/s/{make_share_token(song_id, time.time() + 7 * 86400)}" if song_id and base and (OUTPUTS / song_id / "audio.flac").is_file() else ""
    return f"✅ {what} ready{who}: {job.title}{length}{took}{link}"


def song_family(song_id):
    """Where a song came from and what was made from it (New take / Same score / Edit score / Try styles)."""
    songs = {s["id"]: s for s in library()}
    me = songs.get(song_id)
    parent = songs.get(me["origin"]) if me and me.get("origin") else None
    children = [s for s in songs.values() if s.get("origin") == song_id]
    slim = lambda s: {"id": s["id"], "title": s["title"], "created": s["created"]}
    return {"parent": slim(parent) if parent else None, "children": [slim(s) for s in sorted(children, key=lambda s: s["created"])]}


SHARE_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Music Gen Studio</title>
<style>
  body{margin:0;min-height:100vh;display:grid;place-items:center;font:15px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;color:#e9ebf1;background:#0d0f13}
  main{width:min(520px,92vw);background:#171b22;border:1px solid #272d38;border-radius:14px;padding:26px 24px;box-shadow:0 8px 30px rgba(0,0,0,.35)}
  h1{font-size:20px;margin:0 0 6px;display:flex;align-items:center;gap:10px}
  .mark{width:34px;height:34px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,#f4b543,#ff8a5b);color:#1b1300;font-weight:700;font-size:20px;flex:none}
  .style,.sub{color:#8f97a8;font-size:13px;margin:0 0 14px}.sub{margin:10px 0 2px}
  audio{width:100%;border-radius:999px}
  .btn{display:inline-block;margin-top:8px;padding:9px 14px;border-radius:9px;background:#1e232c;border:1px solid #353d4b;color:#e9ebf1;text-decoration:none;font-size:13px}
  footer{margin-top:16px;color:#5f6675;font-size:12px}
</style></head><body><main>
  <h1><span class="mark">&#9834;</span>{title}</h1>
  {body}
  <footer>Made with Music Gen Studio</footer>
</main></body></html>"""


def run_abc_tool(body):
    """Score helpers: upstream abc_tools.py (strip chords / keep a voice / inspect / compare),
    the local transposer, and a tempo rewrite."""
    import tempfile

    def tool(*args):
        return subprocess.run([sys.executable, "-X", "utf8", str(ABC_TOOLS), *args], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=120)

    if body.action == "transpose":
        from abc_transpose import transpose_abc
        try:
            return {"abc": transpose_abc(body.abc, int(body.semitones))}
        except Exception as exc:
            raise HTTPException(422, f"Could not transpose: {exc}")
    if body.action == "tempo":
        if not body.bpm:
            raise HTTPException(422, "bpm is required")
        new, count = re.subn(r"^(Q:\s*\S+=)\d+", lambda m: f"{m.group(1)}{int(body.bpm)}", body.abc, count=1, flags=re.M)
        if not count:
            raise HTTPException(422, "The score has no Q: tempo line")
        return {"abc": new}
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "score.abc"
        src.write_bytes(body.abc.encode("utf-8"))
        if body.action in ("strip-chords", "keep-voice"):
            voice = body.voice if body.action == "keep-voice" else "both"
            if voice not in ("both", "Vocal", "Ins"):
                raise HTTPException(422, "voice must be both, Vocal or Ins")
            out = Path(tmp) / "out.abc"
            result = tool("strip-chords", str(src), str(out), "--keep-voice", voice)
            if result.returncode:
                raise HTTPException(422, (result.stderr or result.stdout).strip()[-800:] or "abc_tools failed")
            return {"abc": out.read_text(encoding="utf-8")}
        if body.action == "inspect":
            result = tool("inspect", str(src))
            if result.returncode:
                raise HTTPException(422, (result.stderr or result.stdout).strip()[-800:] or "The score does not parse")
            info = json.loads(result.stdout)
            summary = {k: v for k, v in info.items() if k != "voices"}
            summary["voices"] = {name: {k: v for k, v in data.items() if k not in ("notes", "bars", "chords")}
                                 for name, data in info.get("voices", {}).items()}
            return summary
        if body.action == "compare":
            edited = Path(tmp) / "edited.abc"
            edited.write_bytes((body.abc2 or "").encode("utf-8"))
            args = ["compare", str(src), str(edited)] + (["--allow-tempo-change"] if body.allow_tempo_change else [])
            result = tool(*args)
            return {"ok": result.returncode == 0, "report": (result.stdout + result.stderr).strip()[-4000:]}
    raise HTTPException(422, "Unknown action")


LOGIN_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Music Gen Studio</title>
<style>
  body{margin:0;min-height:100vh;display:grid;place-items:center;font:15px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;color:#e9ebf1;background:#0d0f13}
  form{width:min(360px,92vw);background:#171b22;border:1px solid #272d38;border-radius:14px;padding:26px 24px;box-shadow:0 8px 30px rgba(0,0,0,.35)}
  h1{font-size:18px;margin:0 0 4px;display:flex;align-items:center;gap:10px}
  .mark{width:34px;height:34px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,#f4b543,#ff8a5b);color:#1b1300;font-weight:700;font-size:20px}
  p{color:#8f97a8;margin:0 0 18px;font-size:13px}
  input{width:100%;box-sizing:border-box;font:inherit;color:#e9ebf1;background:#12151b;border:1px solid #353d4b;border-radius:8px;padding:11px 12px;margin-bottom:12px;outline:none}
  input:focus{border-color:#f4b543;box-shadow:0 0 0 3px rgba(244,181,67,.18)}
  button{width:100%;padding:12px;border:0;border-radius:10px;font:inherit;font-weight:700;color:#1b1300;background:linear-gradient(180deg,#ffd27a,#f4b543);cursor:pointer}
  .err{color:#f07178;background:rgba(240,113,120,.08);border:1px solid rgba(240,113,120,.35);border-radius:8px;padding:8px 10px;font-size:13px;margin-bottom:12px}
</style></head><body>
<form method="post" action="/login" autocomplete="on">
  <h1><span class="mark">&#9834;</span>Music Gen Studio</h1>
  <p>Enter the password to use the studio.</p>
  {error}
  <input type="password" name="password" placeholder="Password" autofocus required>
  <button type="submit">Sign in</button>
</form></body></html>"""


def install_password(app, password: str):
    """Password login: a cookie session set by /login (works in every browser, including in-app
    ones that never show HTTP auth prompts), with HTTP Basic auth accepted as well for scripts."""
    import asyncio
    import base64
    import hashlib
    import hmac
    import secrets
    from fastapi import Form
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import RedirectResponse, Response as StarletteResponse

    secret = hashlib.sha256(("music-gen-studio:" + password).encode("utf-8")).digest()
    session = hmac.new(secret, b"session", hashlib.sha256).hexdigest()   # stable until the password changes
    cookie = "mgs_session"

    def password_ok(supplied):
        return secrets.compare_digest(supplied.encode("utf-8"), password.encode("utf-8"))

    def authorized(request):
        if secrets.compare_digest(request.cookies.get(cookie, ""), session):
            return True
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                return password_ok(base64.b64decode(header[6:]).decode("utf-8").split(":", 1)[-1])
            except (ValueError, UnicodeDecodeError):
                return False
        return False

    @app.get("/login", response_class=HTMLResponse)
    def login_form():
        return LOGIN_PAGE.replace("{error}", "")

    @app.post("/login")
    async def login(password_field: str = Form(alias="password")):
        if not password_ok(password_field):
            await asyncio.sleep(0.8)                       # slow down guessing
            return HTMLResponse(LOGIN_PAGE.replace("{error}", '<div class="err">Wrong password.</div>'), status_code=401)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(cookie, session, max_age=60 * 60 * 24 * 30, httponly=True, samesite="lax", path="/")
        return response

    @app.get("/logout")
    def logout():
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(cookie, path="/")
        return response

    class PasswordMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            path = request.url.path
            if path == "/login" or path.startswith("/s/") or authorized(request):   # /s/ = signed listen-only links
                return await call_next(request)
            wants_page = request.method == "GET" and "text/html" in request.headers.get("accept", "")
            if wants_page:
                return RedirectResponse("/login", status_code=303)
            return StarletteResponse("Music Gen Studio: password required (sign in at /login)", status_code=401,
                                     headers={"WWW-Authenticate": 'Basic realm="Music Gen Studio", charset="UTF-8"'})

    app.add_middleware(PasswordMiddleware)


CLOUDFLARED_CANDIDATES = [Path(r"C:\Program Files (x86)\cloudflared\cloudflared.exe"),
                          Path(r"C:\Program Files\cloudflared\cloudflared.exe"), HERE / "tools" / "cloudflared.exe"]


def find_cloudflared(explicit=None):
    if explicit:
        return explicit if Path(explicit).is_file() else None
    found = shutil.which("cloudflared")
    if found:
        return Path(found)
    return next((c for c in CLOUDFLARED_CANDIDATES if c.is_file()), None)


class Tunnel:
    """Cloudflare quick tunnel to the local server; the public URL goes to the share pill."""

    URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

    RESTART_DELAY = 10          # seconds; cloudflared reconnects by itself, this covers it exiting outright

    def __init__(self, exe, port, engine):
        self.exe, self.port, self.engine, self.url = exe, port, engine, None
        self.process, self.closing = None, False
        atexit.register(self.stop)
        threading.Thread(target=self._supervise, name="cloudflared", daemon=True).start()

    def _start(self):
        print("  Starting Cloudflare tunnel ...", file=sys.stderr)
        self.process = subprocess.Popen([str(self.exe), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace")

    def _supervise(self):
        while not self.closing:
            self._start()
            self._pump()
            if self.closing:
                break
            print(f"  [cloudflared] exited; restarting in {self.RESTART_DELAY}s (the public address will change)", file=sys.stderr)
            time.sleep(self.RESTART_DELAY)

    def _pump(self):
        announced = None
        for line in self.process.stdout:
            match = self.URL.search(line)
            if match and announced is None:
                announced = self.url = match.group(0)
                self.engine.share_urls = [self.url] + [u for u in getattr(self.engine, "share_urls", []) if u != self.url]
                (HERE / "public_url.txt").write_text(self.url + "\n", encoding="utf-8")
                print(f"\n  Public address (Cloudflare Tunnel) -> {self.url}\n"
                      "  Open the page and click the share pill for a QR code; the password applies there too.\n",
                      file=sys.stderr)
            elif " ERR " in line or "error" in line.lower():
                print("  [cloudflared] " + line.strip()[:160], file=sys.stderr)
        if announced is None:
            print("  [cloudflared] exited before giving a public address", file=sys.stderr)

    def stop(self):
        self.closing = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()


def keep_awake():
    """Ask Windows not to sleep while the server runs (the display may still turn off)."""
    if sys.platform != "win32":
        return False
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

    def hold():
        while True:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            time.sleep(60)
    ok = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0
    if ok:
        threading.Thread(target=hold, name="keep-awake", daemon=True).start()
    return ok


def lan_addresses():
    """IPv4 addresses of this machine's real interfaces, the default-route one first."""
    import socket
    found = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith(("127.", "169.254.")) and ip not in found:
                found.append(ip)
    except OSError:
        pass
    try:                                     # no packet is sent; this only resolves the outbound route
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        primary = probe.getsockname()[0]
        probe.close()
        if primary in found:
            found.remove(primary)
        if not primary.startswith(("127.", "169.254.")):
            found.insert(0, primary)
    except OSError:
        pass
    return found


def open_when_ready(host, port, url, timeout=60):
    """Open the page in the default browser once the server accepts connections."""
    import socket
    import webbrowser
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.3)
    webbrowser.open(url)


if __name__ == "__main__":
    main()
