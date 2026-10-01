#!/usr/bin/env bash
# Installs Docker Engine + NVIDIA Container Toolkit on Ubuntu 26.04 (resolute).
# Run with: sudo bash linux/install-docker.sh
#   --no-nvidia   AMD / ROCm box: Docker Engine only (no NVIDIA repository,
#                 toolkit or runtime); the containers reach the card through
#                 /dev/kfd + /dev/dri instead (compose.rocm.yml).
set -euo pipefail

NVIDIA=1
for a in "$@"; do case $a in --no-nvidia) NVIDIA=0 ;; *) echo "unknown option: $a" >&2; exit 1 ;; esac; done
REAL_USER="${SUDO_USER:-jrlabanza}"
echo "==> Installing for user: $REAL_USER$( (( NVIDIA )) || echo ' (no NVIDIA toolkit)')"

echo "==> [1/6] Prerequisites"
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg git

echo "==> [2/6] Docker apt repository"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes
chmod a+r /etc/apt/keyrings/docker.gpg
cat > /etc/apt/sources.list.d/docker.list <<LIST
deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu resolute stable
LIST

if (( NVIDIA )); then
  echo "==> [3/6] NVIDIA Container Toolkit apt repository"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg --yes
  chmod a+r /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
else
  echo "==> [3/6] NVIDIA Container Toolkit: skipped (--no-nvidia)"
fi

echo "==> [4/6] Installing packages"
apt-get update -qq
PKGS=(docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin)
(( NVIDIA )) && PKGS+=(nvidia-container-toolkit)
apt-get install -y "${PKGS[@]}"

if (( NVIDIA )); then
  echo "==> [5/6] Wiring the NVIDIA runtime into Docker"
  nvidia-ctk runtime configure --runtime=docker
else
  echo "==> [5/6] Starting Docker (AMD: no container runtime needed, the amdgpu driver is in the kernel)"
fi
systemctl enable --now docker
systemctl restart docker

echo "==> [6/6] Adding $REAL_USER to the docker group"
usermod -aG docker "$REAL_USER"

echo
echo "===================== VERIFY ====================="
docker --version
docker compose version
if (( NVIDIA )); then
  echo "--- GPU passthrough test ---"
  if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi 2>&1 | tail -n +2 | head -12; then
    echo "GPU PASSTHROUGH: OK"
  else
    echo "GPU PASSTHROUGH: FAILED - see output above"
  fi
else
  echo "--- AMD device nodes (the ROCm container maps these) ---"
  ls -l /dev/kfd /dev/dri 2>&1 | head -8
fi
echo "=================================================="
echo
echo "IMPORTANT: the docker group only applies to NEW logins."
echo "Run 'newgrp docker' in your shell, or log out and back in."
