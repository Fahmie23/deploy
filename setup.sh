#!/bin/bash
set -e

echo "================================================"
echo "  Singing Judge - WSL2 Setup Script"
echo "================================================"
echo ""

# ── Step 1: Fix Docker credential helper ─────────────────────────────────────
echo "[1/4] Fixing Docker credential helper..."
mkdir -p ~/.docker
cat > ~/.docker/config.json << 'EOF'
{}
EOF
echo "      Done."

# ── Step 2: Install NVIDIA Container Toolkit ─────────────────────────────────
echo "[2/4] Installing NVIDIA Container Toolkit..."

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update -qq
sudo apt-get install -y nvidia-container-toolkit
echo "      Done."

# ── Step 3: Configure Docker to use NVIDIA runtime ───────────────────────────
echo "[3/4] Configuring Docker NVIDIA runtime..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
echo "      Done."

# ── Step 4: Verify GPU is accessible ─────────────────────────────────────────
echo "[4/4] Verifying GPU access..."
if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "      GPU detected successfully."
else
    echo ""
    echo "  WARNING: GPU not detected. Make sure:"
    echo "    - NVIDIA drivers are installed on Windows"
    echo "    - Docker Desktop WSL2 integration is enabled"
    echo ""
fi

echo ""
echo "================================================"
echo "  Setup complete! Run the stack with:"
echo "    docker compose up -d"
echo "================================================"
