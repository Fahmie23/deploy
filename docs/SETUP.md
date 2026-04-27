# VSING Singing Judge — Deployment Guide

## First-Time Setup (Fresh Machine)

Follow these steps once before you can use the API for the first time.

### 1. Open WSL2

Press **Win + R** and type:

```
wsl -d Ubuntu-22.04
```

---

### 2. Install Docker Engine in WSL

Run the following inside WSL — this installs Docker natively without Docker Desktop:

```bash
# Remove any old installs
sudo apt remove docker docker-engine docker.io containerd runc

# Install via official script
curl -fsSL https://get.docker.com | sh

# Add your user to the docker group (avoids needing sudo for every docker command)
sudo usermod -aG docker $USER
newgrp docker
```

Then configure Docker to start automatically whenever WSL launches:

```bash
echo 'sudo service docker start > /dev/null 2>&1' >> ~/.bashrc
```

---

### 3. Install the NVIDIA Container Toolkit

Required so Docker can access your GPU:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
```

Verify the GPU is accessible from Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

### 4. Copy the project into WSL

> Running the project directly from `/mnt/c/...` is slow and causes file permission issues. Copy it into the WSL home directory instead.

```bash
cp -r /mnt/c/Users/fahmi/Downloads/deploy ~/deploy
cd ~/deploy
```

---

### 5. Run the setup script

```bash
bash setup.sh
```

This does the following:
- Fixes the Docker credential helper for WSL2
- Verifies that the GPU is accessible from Docker

> This only needs to be run **once** on a fresh machine. Skip this on subsequent deployments.

---

### 6. Start the API

```bash
docker compose up -d
```

The first run will take **5–10 minutes** — the AI models are being downloaded and loaded. Subsequent starts are much faster.

---

### 7. Verify the API is ready

```bash
curl http://localhost/vsing_api/stats
```

You should see a JSON response like:

```json
{
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "allocated_mb": 312.5,
  "reserved_mb": 512.0,
  "total_mb": 12288.0,
  "free_mb": 11776.0
}
```

If you get a connection error, wait 30–60 seconds and try again — the models are still loading in the background.

---

## Subsequent Starts (Already Set Up)

Once the machine has been set up, you only need to do this every time you want to use the API:

1. Open WSL2: **Win + R** → `wsl -d Ubuntu-22.04`
2. Navigate to the deploy folder: `cd ~/deploy`
3. Start the API: `docker compose up -d`
4. Verify: `curl http://localhost/vsing_api/stats`

> Docker starts automatically when WSL launches (configured in step 2). No need to open Docker Desktop.

---

## Stopping the API

```bash
docker compose down
```

---

## Useful Commands

| Task | Command |
|------|---------|
| Start the API | `docker compose up -d` |
| Stop the API | `docker compose down` |
| Check container status | `docker compose ps` |
| View logs | `docker compose logs -f singing-judge` |
| Rebuild after code changes | `docker compose up --build --force-recreate` |
| Check Docker service status | `sudo service docker status` |
| Start Docker manually | `sudo service docker start` |
