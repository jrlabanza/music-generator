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

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      ca-certificates curl git git-lfs wget \
      build-essential ninja-build pkg-config cmake \
      ffmpeg libsndfile1 sox libsox-fmt-all \
      libgl1 libglib2.0-0 libgomp1 \
      libjpeg-turbo8 libpng16-16; \
    rm -rf /var/lib/apt/lists/*; \
    python3.12 --version

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

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
