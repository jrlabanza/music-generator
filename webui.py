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
import json
import mimetypes
import queue
import random
import re
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import run_lowvram
from run_lowvram import HERE, MODELS, REPO, LowVRAMPipeline

WEB = HERE / "webui"
OUTPUTS = HERE / "outputs"
EXAMPLES = REPO / "examples"
TOKENS_PER_SECOND = 25            # semantic tokens per second of audio (25 Hz latents)
mimetypes.add_type("audio/flac", ".flac")


# ── Pipeline with observable stages ───────────────────────────────────────────
class WebPipeline(LowVRAMPipeline):
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
    created: float = field(default_factory=time.time)
    state: str = "queued"              # queued | running | done | failed | cancelled
    error: str | None = None
    started: float | None = None
    finished: float | None = None
    cancel_requested: bool = False

    def public(self):
        return {"id": self.id, "title": self.title, "state": self.state, "error": self.error,
                "created": self.created, "started": self.started, "finished": self.finished,
                "request": self.request}


class Engine:
    """One worker thread owns the pipeline and all CUDA work."""

    def __init__(self, args):
        self.args = args
        self.pipe: WebPipeline | None = None
        self.model_state, self.model_error = "loading", None
        self.queue: "queue.Queue[Job]" = queue.Queue()
        self.jobs: dict[str, Job] = {}
        self.current: Job | None = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="yue2-worker", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.pipe = WebPipeline.from_pretrained(
                self.args.model, vae=self.args.vae, device="cuda", memory_budget_gib=8,
                backend="torch-eager" if self.args.quantization == "fp8" else "torch",
                quantization=self.args.quantization, gpu_reserve_gib=self.args.gpu_reserve_gib)
            self.pipe.preload()
            self.model_state = "ready"
        except Exception as exc:                       # surface to the page instead of dying silently
            self.model_state, self.model_error = "error", f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            return
        while True:
            job = self.queue.get()
            if job.cancel_requested:
                job.state, job.finished = "cancelled", time.time()
                continue
            self._generate(job)

    def _generate(self, job):
        with self.lock:
            self.current, job.state, job.started = job, "running", time.time()
        out = OUTPUTS / job.id
        try:
            song = self.pipe(**job.request, cancelled=lambda: job.cancel_requested)
            song.save_artifacts(out)
            (out / "title.txt").write_text(job.title, encoding="utf-8")
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
            with self.lock:
                self.current = None

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
    def submit(self, title, request):
        job = Job(id=unique_id(title), title=title, request=request)
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
                          "quantization": self.args.quantization},
                "gpu": gpu, "current": current, "queue": queued}


# ── Output library ────────────────────────────────────────────────────────────
def slugify(text, fallback="song"):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:40].strip("-")
    return slug or fallback


def unique_id(title):
    base = f"{datetime.now():%Y%m%d-%H%M%S}-{slugify(title)}"
    candidate, n = base, 2
    while (OUTPUTS / candidate).exists():
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
    if result is None or request is None or not (directory / "audio.flac").is_file():
        return None
    title_file = directory / "title.txt"
    title = title_file.read_text(encoding="utf-8").strip() if title_file.is_file() else request.get("id", directory.name)
    return {"id": directory.name, "title": title, "style": request.get("style", ""),
            "cot": request.get("cot", "full"), "seed": request.get("seed"),
            "seconds": result.get("audio_seconds"), "truncated": result.get("truncated"),
            "created": (directory / "result.json").stat().st_mtime,
            "has_score": (directory / "score.abc").is_file(),
            "elapsed": (result.get("timing") or {}).get("e2e_seconds")}


def library():
    songs = [s for d in OUTPUTS.iterdir() if d.is_dir() and (s := song_summary(d))] if OUTPUTS.is_dir() else []
    return sorted(songs, key=lambda s: s["created"], reverse=True)


# ── HTTP ──────────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    title: str = ""
    style: str
    lyrics: str
    cot: str = "full"
    seed: int | None = None
    abc: str | None = None
    cfg_scale: float | None = Field(default=None, ge=0, le=20)


def build_app(engine: Engine):
    app = FastAPI(title="YuE2 Studio")
    OUTPUTS.mkdir(exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB / "index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    def status():
        return engine.status()

    @app.get("/api/examples")
    def examples():
        song = read_json(EXAMPLES / "song.json") or {}
        scores = {name: (EXAMPLES / f"{name}.abc").read_text(encoding="utf-8")
                  for name in ("melody", "score", "score-jazz") if (EXAMPLES / f"{name}.abc").is_file()}
        return {"song": song, "scores": scores}

    @app.post("/api/generate")
    def generate(body: GenerateRequest):
        if engine.model_state == "error":
            raise HTTPException(503, f"Model failed to load: {engine.model_error}")
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
        job = engine.submit(title, request)
        return job.public()

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
                "score": score, "audio_url": f"/outputs/{song_id}/audio.flac"}

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
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from other devices on your network")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", default=str(MODELS / "YuE2-3B"))
    parser.add_argument("--vae", default=str(MODELS / "YuE2-Vae"))
    parser.add_argument("--quantization", choices=("none", "fp8"), default="none")
    parser.add_argument("--gpu-reserve-gib", type=float, default=2.0)
    parser.add_argument("--graph-attention", choices=("cudnn", "sdpa"), default="cudnn")
    parser.add_argument("--open", action="store_true", help="open the page in your browser once the server is up")
    args = parser.parse_args()
    for path in (args.model, args.vae):
        if not Path(path).is_dir():
            sys.exit(f"Model directory not found: {path} -- run download_models.py first (about 7.3 GB).")
    run_lowvram.GRAPH_ATTENTION = args.graph_attention
    import uvicorn
    engine = Engine(args)
    print(f"\n  YuE2 Studio -> http://{args.host}:{args.port}\n", file=sys.stderr)
    uvicorn.run(build_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
