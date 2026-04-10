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

# docker entrypoint already ran apt-get update + git clone/pull;
# we only need openssh-server (not in base image) for SSH access.
apt-get install -y -qq openssh-server
# configure and start sshd so we can SSH in for log monitoring
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHuwG1CyywlisyukBnpx0iMb+zRiN6l4rzeg8I1W7pSg tsuberim@gmail.com" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
echo "PasswordAuthentication no" >> /etc/ssh/sshd_config
/usr/sbin/sshd

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
export HF_HOME=/workspace/hf_cache

cat > .env <<EOF
HF_REPO=${HF_REPO}
WANDB_API_KEY=${WANDB_API_KEY}
EOF

# ── data ──────────────────────────────────────────────────────────────────────
if [ ! -d "/workspace/data/tokenized" ] || [ -z "$(ls /workspace/data/tokenized/shard_*.bin 2>/dev/null)" ]; then
    echo "→ ERROR: tokenized shards not found at /workspace/data/tokenized"
    echo "   Upload the dataset to the volume before launching a training pod."
    exit 1
else
    echo "→ data shards found"
fi

# ── train (5 hour safety cutoff) ──────────────────────────────────────────────
echo "→ starting training (${TRAIN_SCRIPT:-train_7b.sh}, 5h limit)..."
timeout 5h bash "${TRAIN_SCRIPT:-train_7b.sh}" 2>&1 | tee /workspace/train.log
EXIT=${PIPESTATUS[0]}

if [ $EXIT -eq 0 ] || [ $EXIT -eq 124 ]; then
    echo "→ training done/timeout (code $EXIT) — terminating pod."
    curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
      -H "Content-Type: application/json" \
      -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID:-}\\\"}) }\"}" || true
else
    echo "→ training crashed (code $EXIT) — pod kept alive for debugging. SSH in and check /workspace/train.log"
    sleep infinity
fi
