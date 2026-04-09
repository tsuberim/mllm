#!/bin/bash
# RunPod container startup script for merlin training.
#
# Required RunPod environment variables:
#   GITHUB_TOKEN   — personal access token with repo read access
#   HF_REPO        — e.g. tsuberim/merlin
#   WANDB_API_KEY  — weights & biases API key
#
# Recommended pod config:
#   GPU:  1× H100 SXM 80 GB
#   Disk: 50 GB (repo + data + checkpoints)
#   Image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

set -euo pipefail

apt-get update -qq && apt-get install -y -qq git curl openssh-server
# configure and start sshd so we can SSH in for log monitoring
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHuwG1CyywlisyukBnpx0iMb+zRiN6l4rzeg8I1W7pSg tsuberim@gmail.com" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
echo "PasswordAuthentication no" >> /etc/ssh/sshd_config
/usr/sbin/sshd

REPO_URL="https://${GITHUB_TOKEN}@github.com/tsuberim/merlin.git"
WORKDIR="/workspace/merlin"

# ── clone / update repo ───────────────────────────────────────────────────────
if [ ! -d "$WORKDIR/.git" ]; then
    echo "→ cloning repo..."
    git clone "$REPO_URL" "$WORKDIR"
else
    echo "→ pulling latest..."
    git -C "$WORKDIR" pull
fi
cd "$WORKDIR"

# ── python deps ───────────────────────────────────────────────────────────────
if [ ! -f ".deps_installed" ]; then
    echo "→ installing dependencies..."
    pip install --quiet --upgrade pip
    # skip mlx (Mac-only), torch (pre-installed in image), pytest (not needed)
    grep -vE "^(mlx|torch|pytest)" requirements.txt | pip install --quiet -r /dev/stdin
    touch .deps_installed
else
    echo "→ deps already installed"
fi

# ── env ───────────────────────────────────────────────────────────────────────
cat > .env <<EOF
HF_REPO=${HF_REPO}
WANDB_API_KEY=${WANDB_API_KEY}
EOF

# ── data ──────────────────────────────────────────────────────────────────────
if [ ! -f "data_train.bin" ]; then
    echo "→ tokenizing TinyStories (~5 min)..."
    python data.py
else
    echo "→ data already prepared"
fi

# ── train (5 hour safety cutoff) ──────────────────────────────────────────────
echo "→ starting training (${TRAIN_SCRIPT:-train_macbook.sh}, 5h limit)..."
timeout 5h bash "${TRAIN_SCRIPT:-train_macbook.sh}" 2>&1 | tee /workspace/train.log
EXIT=${PIPESTATUS[0]}

if [ $EXIT -eq 124 ]; then
    echo "→ 5h limit reached — terminating pod. Resume manually when ready."
else
    echo "→ training exited (code $EXIT) — terminating pod."
fi

# ── terminate pod to stop billing ─────────────────────────────────────────────
curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}"
