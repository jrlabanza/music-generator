#!/usr/bin/env python
"""Transpose a YuE2 two-voice ABC score by a number of semitones.

    python abc_transpose.py score.abc +4 --key Am -o score-am.abc

Handles the limited dialect YuE2 emits: a K: header, quoted chord symbols
("Fm", "Db/F", "Bbm7"), notes with ABC accidentals and octave marks, rests
(z, Z), durations, ties and bar lines. Key-signature accidentals and
measure-scoped explicit accidentals are resolved before shifting, and the
result is respelled against the target key signature. Verify with
`YuE/skills/yue2-music/scripts/abc_tools.py inspect` (every midi_pitch
should move by the same amount).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

NOTE_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"
# fifths position of major keys; relative minors are three fifths flatter
MAJOR_FIFTHS = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
                "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7}
TOKEN = re.compile(r'"[^"]*"|\[[^\]]*\]|[_^=]*[A-Ga-g][,\']*|[zZ]|\d+/\d+|/\d*|\d+|.', re.S)


def key_signature(key: str) -> dict[str, int]:
    """Map note letters to the accidental (+1/-1) implied by a key such as Fm or Db."""
    key = key.strip()
    minor = key.endswith("m") and not key.endswith("dim")
    tonic = key[:-1] if minor else key
    if minor:
        idx = FLAT_NAMES.index(tonic) if tonic in FLAT_NAMES else SHARP_NAMES.index(tonic)
        major = (idx + 3) % 12
        candidates = [n for n in (SHARP_NAMES[major], FLAT_NAMES[major]) if n in MAJOR_FIFTHS]
        fifths = min((MAJOR_FIFTHS[c] for c in candidates), key=abs)
    else:
        if tonic not in MAJOR_FIFTHS:
            raise ValueError(f"Unsupported key {key}")
        fifths = MAJOR_FIFTHS[tonic]
    if fifths >= 0:
        return {letter: 1 for letter in SHARP_ORDER[:fifths]}
    return {letter: -1 for letter in FLAT_ORDER[:-fifths]}


def spell_key(key: str, semitones: int) -> str:
    minor = key.endswith("m") and not key.endswith("dim")
    tonic = key[:-1] if minor else key
    idx = (FLAT_NAMES.index(tonic) if tonic in FLAT_NAMES else SHARP_NAMES.index(tonic)) + semitones
    idx %= 12
    if minor:
        names = [n + "m" for n in (SHARP_NAMES[idx], FLAT_NAMES[idx]) if n in MAJOR_FIFTHS or n in ("A#", "D#", "G#", "Ab", "Eb", "Bb")]
        return names[0] if names else SHARP_NAMES[idx] + "m"
    return SHARP_NAMES[idx] if MAJOR_FIFTHS.get(SHARP_NAMES[idx], 99) != 99 and abs(MAJOR_FIFTHS.get(SHARP_NAMES[idx], 99)) <= abs(MAJOR_FIFTHS.get(FLAT_NAMES[idx], 99)) else FLAT_NAMES[idx]


def transpose_chord(symbol: str, semitones: int, prefer_flats: bool) -> str:
    names = FLAT_NAMES if prefer_flats else SHARP_NAMES

    def shift(match):
        root = match.group(0)
        idx = (FLAT_NAMES.index(root) if root in FLAT_NAMES else SHARP_NAMES.index(root)) + semitones
        return names[idx % 12]
    return re.sub(r"[A-G][#b]?", shift, symbol)


class Transposer:
    def __init__(self, source_key: str, semitones: int, target_key: str | None = None):
        self.semitones = semitones
        self.source_sig = key_signature(source_key)
        self.target_key = target_key or spell_key(source_key, semitones)
        self.target_sig = key_signature(self.target_key)
        self.prefer_flats = any(v < 0 for v in self.target_sig.values())
        self.measure_accidentals: dict[tuple[str, int], int] = {}

    def note_to_midi(self, accidental: str, letter: str, octave_marks: str) -> int:
        octave = 5 if letter.islower() else 4
        octave += octave_marks.count("'") - octave_marks.count(",")
        upper = letter.upper()
        key_tuple = (upper, octave)
        if accidental:
            alter = {"^": 1, "^^": 2, "_": -1, "__": -2, "=": 0}[accidental]
            self.measure_accidentals[key_tuple] = alter
        elif key_tuple in self.measure_accidentals:
            alter = self.measure_accidentals[key_tuple]
        else:
            alter = self.source_sig.get(upper, 0)
        return 12 * (octave + 1) + NOTE_SEMITONE[upper] + alter

    def midi_to_abc(self, midi: int) -> str:
        octave, pitch_class = divmod(midi, 12)
        octave -= 1
        names = FLAT_NAMES if self.prefer_flats else SHARP_NAMES
        name = names[pitch_class]
        letter, alter = name[0], {"#": 1, "b": -1}.get(name[1:], 0)
        expected = self.target_sig.get(letter, 0)
        accidental = "" if alter == expected else {1: "^", -1: "_", 0: "="}[alter]
        # measure-scoped: once an explicit accidental is written, later naturals in the bar need "="
        key_tuple = (letter, octave)
        if accidental:
            self.target_measure[key_tuple] = alter
        elif key_tuple in self.target_measure and self.target_measure[key_tuple] != expected:
            accidental = "="
            self.target_measure[key_tuple] = expected
        if octave >= 5:
            text = letter.lower() + "'" * (octave - 5)
        else:
            text = letter + "," * (4 - octave)
        return accidental + text

    def transpose_line(self, line: str) -> str:
        out = []
        self.measure_accidentals = {}
        self.target_measure: dict[tuple[str, int], int] = {}
        for token in TOKEN.findall(line):
            if token.startswith('"'):
                inner = token[1:-1]
                out.append('"' + (transpose_chord(inner, self.semitones, self.prefer_flats) if re.match(r"[A-G]", inner) else inner) + '"')
            elif token == "|":
                self.measure_accidentals = {}
                self.target_measure = {}
                out.append(token)
            elif re.fullmatch(r"[_^=]*[A-Ga-g][,']*", token):
                accidental = re.match(r"[_^=]*", token).group(0)
                letter = token[len(accidental)]
                marks = token[len(accidental) + 1:]
                out.append(self.midi_to_abc(self.note_to_midi(accidental, letter, marks) + self.semitones))
            else:
                out.append(token)
        return "".join(out)


def transpose_abc(text: str, semitones: int, target_key: str | None = None) -> str:
    lines = text.splitlines()
    source_key = next(l[2:].strip() for l in lines if l.startswith("K:"))
    engine = Transposer(source_key, semitones, target_key)
    out = []
    for line in lines:
        if line.startswith("K:"):
            out.append("K:" + engine.target_key)
        elif line.startswith(("%", "V:", "X:", "T:", "M:", "L:", "Q:")) or not line.strip():
            out.append(line)
        else:
            out.append(engine.transpose_line(line))
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("semitones", type=int, help="e.g. +4 or -3")
    parser.add_argument("--key", help="target key name to write in K: (default: derived)")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    result = transpose_abc(args.source.read_text(encoding="utf-8"), args.semitones, args.key)
    args.output.write_bytes(result.encode("utf-8"))
    print(f"{args.source} -> {args.output} ({args.semitones:+d} semitones, K:{next(l[2:] for l in result.splitlines() if l.startswith('K:'))})")


if __name__ == "__main__":
    main()
