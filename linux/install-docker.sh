#!/usr/bin/env bash
# Installs Docker Engine + NVIDIA Container Toolkit on Ubuntu 26.04 (resolute).
# Run with: sudo bash /home/jrlabanza/AI/bin/00-install-docker.sh
set -euo pipefail

REAL_USER="${SUDO_USER:-jrlabanza}"
echo "==> Installing for user: $REAL_USER"

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

echo "==> [3/6] NVIDIA Container Toolkit apt repository"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg --yes
chmod a+r /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "==> [4/6] Installing packages"
apt-get update -qq
apt-get install -y docker-ce docker-ce-cli containerd.io \
                   docker-buildx-plugin docker-compose-plugin \
                   nvidia-container-toolkit

echo "==> [5/6] Wiring the NVIDIA runtime into Docker"
nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker

echo "==> [6/6] Adding $REAL_USER to the docker group"
usermod -aG docker "$REAL_USER"

echo
echo "===================== VERIFY ====================="
docker --version
docker compose version
echo "--- GPU passthrough test ---"
if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi 2>&1 | tail -n +2 | head -12; then
  echo "GPU PASSTHROUGH: OK"
else
  echo "GPU PASSTHROUGH: FAILED - see output above"
fi
echo "=================================================="
echo
echo "IMPORTANT: the docker group only applies to NEW logins."
echo "Run 'newgrp docker' in your shell, or log out and back in."
