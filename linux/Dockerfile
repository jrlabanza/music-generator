# syntax=docker/dockerfile:1
# Self-contained image: CUDA 13.0 devel (host driver 595 / CUDA 13.2; nvcc is
# needed to build SageAttention/insightface) + Python 3.12, the interpreter
# this tool was running on Windows. Everything below the base block is the
# tool's own environment; code and models are bind-mounted at /app at runtime.
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Survive transient DNS/network hiccups instead of failing the whole build.
RUN printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
      > /etc/apt/apt.conf.d/99-retries

# Python 3.11 (for the voice and SheetSage2 helper environments, which pin an
# older torch) comes from the deadsnakes PPA, added as a STATIC apt source. The
# signing key is fetched by fingerprint and re-verified before use.
ARG DEADSNAKES_FPR=F23C5A6CF475977595C89F51BA6932366A755776
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg; \
    curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&options=mr&exact=on&search=0x${DEADSNAKES_FPR}" \
      -o /tmp/deadsnakes.asc; \
    gpg --show-keys --with-colons /tmp/deadsnakes.asc \
      | awk -F: '$1=="fpr"{print $10}' | grep -qx "${DEADSNAKES_FPR}"; \
    gpg --dearmor < /tmp/deadsnakes.asc > /usr/share/keyrings/deadsnakes.gpg; \
    rm /tmp/deadsnakes.asc; \
    echo "deb [signed-by=/usr/share/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu noble main" \
      > /etc/apt/sources.list.d/deadsnakes.list

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      python3.11 python3.11-venv python3.11-dev \
      ca-certificates curl git git-lfs wget \
      build-essential ninja-build pkg-config cmake \
      ffmpeg libsndfile1 sox libsox-fmt-all \
      libgl1 libglib2.0-0 libgomp1 \
      libjpeg-turbo8 libpng16-16; \
    rm -rf /var/lib/apt/lists/*; \
    python3.12 --version; python3.11 --version

# Writable cache roots, bind-mounted from the host so model downloads survive
# image rebuilds and are shared between tools.
ENV HF_HOME=/cache/hf \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache/xdg \
    MPLCONFIGDIR=/cache/mpl
RUN mkdir -p /cache/hf /cache/torch /cache/xdg /cache/mpl /app \
    && chmod -R 777 /cache

# RTX 3070 = sm_86. Narrowing this keeps any source build fast and small.
ENV TORCH_CUDA_ARCH_LIST="8.6" \
    CUDA_MODULE_LOADING=LAZY

WORKDIR /app

# ---------------------------------------------------------------------------
# yue2 environment
# ---------------------------------------------------------------------------
# YuE 2 / Music Gen Studio - reproduces the working Windows environment:
#   Python 3.12.10, torch 2.10.0+cu128, transformers 4.57.6, numpy 2.2.6
# Note transformers 4.57.6 here vs 4.57.3 in qwen-tts: these pins are mutually
# exclusive, which is precisely why each tool gets its own container.
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip wheel setuptools

# torch only - neither torchvision nor torchaudio was installed on Windows.
RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# constraints.txt is copied AFTER the torch layer on purpose: torch is the
# largest download by far, and putting the COPY first meant every edit to
# the pin list invalidated the cached torch layer and re-fetched ~5 GB.
COPY constraints.txt /opt/constraints.txt

# yue2-infer's pinned dependency set. The package itself is bind-mounted, so
# YuE/src goes on PYTHONPATH instead of an editable install.
RUN pip install -c /opt/constraints.txt \
      transformers==4.57.6 \
      huggingface-hub==0.36.2 \
      safetensors==0.7.0 \
      tiktoken==0.12.0 \
      numpy==2.2.6 \
      soundfile==0.13.1 \
      accelerate==1.13.0

# webui extras from requirements.txt
RUN pip install -c /opt/constraints.txt \
      "fastapi>=0.115" "uvicorn[standard]>=0.30" \
      hf_xet qrcode python-multipart

# Used by stems.py / voice_convert.py / export_audio.py / sheetsage_transcribe.py
RUN pip install -c /opt/constraints.txt librosa pydub \
  || echo "NOTE: optional audio extras failed; core generation still works"

# Fill in the long tail from the working Windows venv (best effort, constrained
# so it cannot disturb the torch stack). Covers dependencies the curated list
# above misses - notably the 43 Forge extensions / 8 ComfyUI custom node packs.
COPY install-pins.sh /usr/local/bin/install-pins.sh
COPY torch-lock.txt /opt/torch-lock.txt
RUN chmod +x /usr/local/bin/install-pins.sh \
 && install-pins.sh /opt/constraints.txt /opt/torch-lock.txt

COPY pip-each.sh assert-imports.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/pip-each.sh /usr/local/bin/assert-imports.sh

# Fail the build now if anything this tool cannot start without is missing,
# rather than discovering it when the server refuses to boot.
RUN assert-imports.sh torch transformers accelerate safetensors soundfile tiktoken numpy fastapi uvicorn

# ---------------------------------------------------------------------------
# Helper environments - the README's ".venv-voice" and ".venv-sheetsage2"
# ---------------------------------------------------------------------------
# On Windows these are two extra venvs the user creates by hand (README: "Sing
# it in your voice", "Cover a recording"). Here they are part of the image, so
# Export (MP3 / loudness), stems, karaoke sync, voice conversion and covers
# work out of the box. Both pin torch 2.8.0+cu126 on Python 3.11, exactly as
# the README says; the pip cache mount means the 2.5 GB torch wheel is fetched
# once for both. webui.py finds them through MUSICGEN_VOICE_PY /
# MUSICGEN_SHEETSAGE_PY, set by entrypoint.sh.
ENV VOICE_ENV=/opt/venv-voice \
    SHEETSAGE_ENV=/opt/venv-sheetsage2
RUN --mount=type=cache,target=/root/.cache/pip,id=ai-pip-cu126 \
    set -eux; \
    export PIP_NO_CACHE_DIR=false; \
    python3.11 -m venv "$VOICE_ENV"; \
    "$VOICE_ENV/bin/pip" install --upgrade pip wheel setuptools; \
    "$VOICE_ENV/bin/pip" install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126; \
    "$VOICE_ENV/bin/pip" install \
      scipy==1.13.1 librosa==0.10.2 "huggingface-hub>=0.28.1" munch==4.0.0 einops==0.8.0 \
      descript-audio-codec==1.0.0 pydub==0.25.1 transformers==4.46.3 soundfile numpy==1.26.4 \
      hydra-core==1.3.2 pyyaml python-dotenv demucs mutagen lameenc pyloudnorm; \
    "$VOICE_ENV/bin/python" -c "import torch, torchaudio, demucs, lameenc, pyloudnorm, mutagen, transformers, scipy, librosa; assert torch.__version__.startswith('2.8.0'), torch.__version__"

RUN --mount=type=cache,target=/root/.cache/pip,id=ai-pip-cu126 \
    set -eux; \
    export PIP_NO_CACHE_DIR=false; \
    python3.11 -m venv "$SHEETSAGE_ENV"; \
    "$SHEETSAGE_ENV/bin/pip" install --upgrade pip wheel; \
    "$SHEETSAGE_ENV/bin/pip" install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126; \
    "$SHEETSAGE_ENV/bin/pip" install \
      huggingface-hub==0.36.0 transformers==4.45.2 safetensors==0.5.3 numpy==1.24.3 scipy==1.13.1 \
      mir_eval==0.8.2 pretty_midi==0.2.10 mido==1.3.3 setuptools==78.1.1 soundfile; \
    "$SHEETSAGE_ENV/bin/python" -c "import torch, transformers, mir_eval, pretty_midi, mido, soundfile; assert torch.__version__.startswith('2.8.0'), torch.__version__"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
