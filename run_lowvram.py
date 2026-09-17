#!/usr/bin/env python
"""Run YuE2 on an 8 GB GPU under Windows.

Upstream YuE2 assumes a 24 GB card and keeps the whole 6.8 GB BF16
Mixture-of-Transformers model resident. This runner keeps only the half that
is in use on the GPU and parks the rest in system RAM:

    phase                GPU                     CPU
    -------------------  ----------------------  ----------------------------------
    plan / semantic      AR layers, lm_head      NAR layers, embedding table
    NAR prefill          AR layers               NAR layers, lm_head, embedding table
    NAR flow-matching    NAR layers              AR layers, lm_head, embedding table
    VAE decode           VAE                     transformer (upstream)

The 0.7 GiB token-embedding table stays in RAM: rows are gathered on the CPU
and copied over, and the CUDA-graph decoder reads them from a static buffer.

It also works around two gaps in the Windows PyTorch build and one upstream
leak:

  * FlashAttention is not compiled in, so grouped-query SDPA silently falls
    back to the O(n^2) math kernel (about 24 GB at song length). K/V heads are
    expanded instead, which selects the memory-efficient kernel.
  * The CUDA-graph decoder would pick the missing flash entrypoint; it is
    pointed at cuDNN attention instead (not seed-reproducible run to run;
    --graph-attention sdpa is, at ~2.7x slower decoding).
  * The VAE's legacy weight_norm leaves 254 MiB of computed weights on the
    GPU after .to("cpu"); they are released after each decode.

Nothing in the upstream checkout is modified; everything is patched in here.

    python run_lowvram.py --output outputs/first-song
    python run_lowvram.py --request YuE/examples/song.json --cot melody --output outputs/melody
    python run_lowvram.py --quantization fp8 --output outputs/fp8      # smallest footprint, slower
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE / "YuE"
MODELS = HERE / "models"

# ── 1. Attention without grouped-query SDPA ────────────────────────────────────
import yue2.modeling_yue2 as modeling
import yue2.nar as nar
import yue2.cuda_graph as cuda_graph


def _expand_kv(query, key, value):
    if query.shape[1] != key.shape[1]:
        groups = query.shape[1] // key.shape[1]
        key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    return key, value


def sdpa_expanded(query, key, value, *, attn_mask=None, is_causal=False):
    """Drop-in for modeling_yue2.sdpa: expand K/V heads so a fused kernel is used."""
    key, value = _expand_kv(query, key, value)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, is_causal=is_causal)


def attention_expanded(q, k, v, *, causal=False, backend="sdpa", query_chunk_size=None):
    """Drop-in for nar.attention ([tokens, heads, dim] layout) with expanded K/V heads."""
    query = q.transpose(0, 1).unsqueeze(0)
    key = k.transpose(0, 1).unsqueeze(0)
    value = v.transpose(0, 1).unsqueeze(0)
    key, value = _expand_kv(query, key, value)
    block = query_chunk_size or len(q)
    outputs = []
    for start in range(0, len(q), block):
        end = min(start + block, len(q))
        used_key = key[..., :end, :] if causal else key
        used_value = value[..., :end, :] if causal else value
        mask = None
        if causal and start:
            mask = (torch.arange(end, device=q.device)[None, :] <=
                    torch.arange(start, end, device=q.device)[:, None])
        outputs.append(F.scaled_dot_product_attention(
            query[..., start:end, :], used_key, used_value,
            attn_mask=mask, is_causal=causal and start == 0))
    return torch.cat(outputs, dim=-2)[0].transpose(0, 1)


modeling.sdpa = sdpa_expanded
nar.attention = attention_expanded

# ── 2. Embedding table in system RAM ───────────────────────────────────────────
class CPUEmbedding(torch.nn.Module):
    """Drop-in for the 0.7 GiB ``embed_tokens`` table that keeps it in system RAM.

    Rows are gathered on the CPU and copied to the GPU (a few KB per decode
    step), so the numerics are identical. Inside a captured CUDA graph no CPU
    work can run, so ``StaticGraphAR`` fills ``static`` before each replay.
    """

    def __init__(self, embedding, device):
        super().__init__()
        self.table = embedding.weight.detach().to("cpu")      # plain attribute: .to() never moves it
        self.num_embeddings, self.embedding_dim = self.table.shape
        self.static = None
        # GraphAR reads embed_tokens.weight for its device/dtype probe.
        self.weight = torch.empty(0, dtype=self.table.dtype, device=device)

    def rows(self, ids):
        return F.embedding(torch.as_tensor(ids, device="cpu"), self.table)

    def forward(self, ids):
        if self.static is not None:
            return self.static
        return self.rows(ids).to(ids.device)


def release_weight_norm(module):
    """Legacy weight_norm caches its computed ``weight`` as a plain tensor, which
    ``.to("cpu")`` leaves behind on the GPU (254 MiB for the VAE)."""
    if module is None:
        return
    for child in module.modules():
        weight = child.__dict__.get("weight")
        if isinstance(weight, torch.Tensor) and weight.device.type == "cuda":
            child.weight = weight.to("cpu")


# ── 3. CUDA-graph decoder: cuDNN attention, static embedding input ─────────────
GRAPH_ATTENTION = "cudnn"


class StaticGraphAR(cuda_graph.GraphAR):
    """GraphAR that uses cuDNN attention (the Windows wheel lacks the flash
    entrypoint it would pick) and feeds a CPU-resident embedding table."""

    def __init__(self, model, prefixes, max_tokens, *, attention_backend="auto", **kwargs):
        if attention_backend == "auto":
            attention_backend = GRAPH_ATTENTION
        super().__init__(model, prefixes, max_tokens, attention_backend=attention_backend, **kwargs)
        embed = model.model.embed_tokens
        self.embed = embed if isinstance(embed, CPUEmbedding) else None
        self.x0 = None
        if self.embed is not None:
            self.x0 = torch.zeros(self.branches, 1, self.embed.embedding_dim, device=self.device, dtype=self.dtype)

    def prefill(self):
        if self.embed is not None:
            self.embed.static = None                 # prefix rows are gathered on the CPU
        return super().prefill()

    def _capture(self):
        if self.embed is not None:
            self.embed.static = self.x0              # the graph reads the static buffer
        super()._capture()

    def step(self, token):
        if self.embed is not None:
            index = int(token.item()) if isinstance(token, torch.Tensor) else int(token)
            self.x0.copy_(self.embed.rows([index]).to(self.dtype)[None].expand(self.branches, 1, -1))
        return super().step(token)

    def close(self):
        if self.embed is not None:
            self.embed.static = None
        self.x0 = None
        super().close()


cuda_graph.GraphAR = StaticGraphAR

# ── 4. Pipeline with phase-wise placement ──────────────────────────────────────
from yue2.pipeline import YuE2Pipeline, SemanticResult
from yue2.protocol import token_prefixes
from yue2.quantization import prepare_fp8_ar, restore_ar
from yue2.nar import CachedNAR, song_chunks


class LowVRAMPipeline(YuE2Pipeline):
    def __init__(self, *args, gpu_reserve_gib=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_reserve_gib = gpu_reserve_gib
        if gpu_reserve_gib is not None and self.device.type == "cuda":
            total = torch.cuda.get_device_properties(self.device).total_memory
            fraction = (total - gpu_reserve_gib * 2**30) / total
            torch.cuda.set_per_process_memory_fraction(min(max(fraction, 0.1), 1.0), self.device)

    # module groups -----------------------------------------------------------
    def _groups(self):
        m = self._model
        ar, nar_modules = [], []                   # embed_tokens lives in RAM (CPUEmbedding)
        for layer in m.model.layers:
            ar += [layer.input_layernorm, layer.self_attn, layer.post_attention_layernorm, layer.mlp]
            nar_modules += [layer.nar_input_layernorm, layer.nar_self_attn,
                            layer.nar_pre_mlp_layernorm, layer.nar_mlp]
        small = [m.model.norm, m.model.rotary_emb, m.llm2vae, m.vae2llm, m.time_embedder, m.latent_pos_embed]
        return ar, nar_modules, [m.lm_head], small

    def _place(self, *, ar, nar, lm_head):
        ar_modules, nar_modules, head, small = self._groups()
        plan = [(ar_modules, ar), (nar_modules, nar), (head, lm_head), (small, True)]
        for group, on_gpu in plan:          # free first ...
            if not on_gpu:
                for module in group:
                    module.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        for group, on_gpu in plan:          # ... then fill
            if on_gpu:
                for module in group:
                    module.to(self.device)

    def _report(self, label):
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 2**30
            peak = torch.cuda.max_memory_allocated(self.device) / 2**30
            print(f"[lowvram] {label}: {allocated:.2f} GiB on GPU, peak so far {peak:.2f} GiB", file=sys.stderr)

    # overrides --------------------------------------------------------------
    def _load_model(self, for_nar=False):
        if self._model is None:
            from yue2.modeling_yue2 import YuE2ForCausalLM
            with self._status("Loading model into system RAM"):
                start = time.perf_counter()
                self._model = YuE2ForCausalLM.from_pretrained(
                    self.model_dir, local_files_only=True,
                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
                self.load_timing["mot_load_seconds"] = time.perf_counter() - start
        model = self._model
        if not isinstance(model.model.embed_tokens, CPUEmbedding):
            model.model.embed_tokens = CPUEmbedding(model.model.embed_tokens, self.device)
        if for_nar:
            self._place(ar=True, nar=False, lm_head=False)
        else:
            if self.quantization == "fp8" and not getattr(model, "_yue2_fp8_originals", None):
                self._place(ar=False, nar=False, lm_head=False)   # quantize from the CPU copies
                prepare_fp8_ar(model, self.device)
            self._place(ar=True, nar=False, lm_head=True)
        self._report("AR weights resident" if not for_nar else "NAR-prefill weights resident")
        return model

    def synthesize(self, semantic, *, cancelled=None):
        if not isinstance(semantic, SemanticResult):
            raise TypeError("Pass the SemanticResult returned by generate_semantic()")
        plan = semantic.plan
        if token_prefixes(plan.request, self.tokenizer, plan.abc_ids) != plan.prefix:
            raise ValueError("Semantic result does not retain the request's exact prefix")
        if self._model is None:
            self._load_model(for_nar=True)
        if self.quantization != "none":
            self._place(ar=False, nar=False, lm_head=False)   # restore exact BF16 AR weights on the CPU
            restore_ar(self._model)
        model = self._load_model(for_nar=True)
        config = self.generation_config
        chunks = song_chunks(plan.prefix, semantic.tokens, plan.request.seed, config.context)
        output = []
        with self._status("Synthesizing audio", unit="steps") as status:
            report = (lambda done, total: status.update(done, total=total)) if self.progress else None
            for index, chunk in enumerate(chunks):
                if cancelled is not None and cancelled():
                    raise InterruptedError("Cancelled before acoustic prefill")
                self._place(ar=True, nar=False, lm_head=False)
                engine = CachedNAR(model, chunk)                 # AR prefill -> per-layer K/V
                self._report(f"chunk {index + 1}/{len(chunks)} prefilled")
                self._place(ar=False, nar=True, lm_head=False)   # swap halves for the ODE
                try:
                    progress = None
                    if report is not None:
                        def progress(done, total, index=index):
                            report(index * total + done, total * len(chunks))
                    output.append(engine.solve(config.ode_steps, cancelled, on_progress=progress))
                finally:
                    engine.close()
                del engine
                self._report(f"chunk {index + 1}/{len(chunks)} solved")
        return torch.cat(output, dim=0).detach().float().cpu().numpy()

    def decode(self, latents, *, full=False, vae=None):
        try:
            return super().decode(latents, full=full, vae=vae)
        finally:
            release_weight_norm(self._vae)           # upstream's .to("cpu") leaves 254 MiB behind
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


# ── 4. CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--request", type=Path, default=REPO / "examples" / "song.json",
                        help="JSON with style/lyrics/cot/seed (default: the upstream example)")
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--style", help="override the request's style prompt")
    parser.add_argument("--lyrics-file", type=Path, help="override the request's lyrics from a text file")
    parser.add_argument("--abc-file", type=Path, help="use this ABC score instead of planning one")
    parser.add_argument("--cot", choices=("full", "melody", "off"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--model", default=str(MODELS / "YuE2-3B"))
    parser.add_argument("--vae", default=str(MODELS / "YuE2-Vae"))
    parser.add_argument("--quantization", choices=("none", "fp8"), default="none",
                        help="fp8 shrinks the AR weights further but forces the slower eager decoder")
    parser.add_argument("--backend", choices=("torch", "torch-eager"),
                        help="default: torch (CUDA graphs) unless --quantization fp8")
    parser.add_argument("--graph-attention", choices=("cudnn", "sdpa"), default="cudnn")
    parser.add_argument("--gpu-reserve-gib", type=float, default=2.0,
                        help="VRAM left for the driver/desktop; lower it if you close other GPU apps")
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Choose a fresh output directory to retain each version.")
    for path in (args.model, args.vae):
        if not Path(path).is_dir():
            parser.error(f"Model directory not found: {path} -- run download_models.py first (about 7.3 GB).")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    request.pop("title", None)                    # web-app request files carry a display title
    if args.style:
        request["style"] = args.style
    if args.lyrics_file:
        request["lyrics"] = args.lyrics_file.read_text(encoding="utf-8")
    if args.abc_file:
        request["abc"] = args.abc_file.read_text(encoding="utf-8")
    if args.cot:
        request["cot"] = args.cot
    if args.seed is not None:
        request["seed"] = args.seed
    if args.cfg_scale is not None:
        request["cfg_scale"] = args.cfg_scale
    if request.get("abc") is not None and request.get("cot", "full") == "off":
        parser.error("A supplied score requires full or melody mode.")

    global GRAPH_ATTENTION
    GRAPH_ATTENTION = args.graph_attention
    backend = args.backend or ("torch-eager" if args.quantization == "fp8" else "torch")

    with LowVRAMPipeline.from_pretrained(
        args.model, vae=args.vae, device="cuda", memory_budget_gib=8,
        backend=backend, quantization=args.quantization, gpu_reserve_gib=args.gpu_reserve_gib,
    ) as pipe:
        song = pipe(**request)
        song.save_artifacts(args.output)
        print(json.dumps({"audio": str(args.output / "audio.flac"),
                          "seconds": round(len(song.audio) / song.sample_rate, 1),
                          "truncated": song.truncated}))
        return 1 if any(song.truncated.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
