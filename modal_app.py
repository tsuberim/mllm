"""
Modal app for merlin training on H100.

Deploy (needed once, or after requirements.txt changes):
    modal deploy modal_app.py

Run an experiment:
    modal run modal_app.py --commit <sha> [--model experiment] [--max-steps 200]

Upload corpus to Modal volume (one-time):
    modal run modal_app.py::upload_corpus
"""
import os
import re
import subprocess
import sys

import modal

REPO_URL = "https://github.com/tsuberim/mllm.git"

app = modal.App("merlin")

# Deps pre-baked into image — built on cheap CPU infra, not on H100.
# Rebuilds automatically when requirements.txt changes.
# mlx / mlx-lm are Apple-only, skip them.
image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime")
    .apt_install("git", "gcc")
    .pip_install(
        "wandb",
        "huggingface_hub",
        "tokenizers",
        "tqdm",
        "python-dotenv",
        "datasets",
    )
)

# Persistent volume holding corpus_train.bin + corpus_val.bin
corpus_vol = modal.Volume.from_name("merlin-corpus", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data/tokenized": corpus_vol},
    timeout=60 * 60 * 12,  # 12h max
    secrets=[modal.Secret.from_name("merlin")],
)
def train(
    commit: str,
    model: str = "experiment",
    batch_size: int = 256,
    max_steps: int = 20000,
    val_every: int = 500,
    val_steps: int = 10,
    save_every: int = 1000,
    bf16: bool = False,
) -> dict:
    # Clone repo at exact commit — only overhead on H100 (~5s)
    repo_dir = f"/tmp/{commit}"
    subprocess.run(
        ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", REPO_URL, repo_dir],
        check=True,
    )
    subprocess.run(
        ["git", "-C", repo_dir, "checkout", "--quiet", "--detach", commit],
        check=True,
    )
    print(f"[train] checked out {commit[:12]}")

    cmd = [
        sys.executable, "-u", f"{repo_dir}/train.py",
        "--model",      model,
        "--batch_size", str(batch_size),
        "--max_steps",  str(max_steps),
        "--val_every",  str(val_every),
        "--val_steps",  str(val_steps),
        "--save_every", str(save_every),
    ]
    if bf16:
        cmd.append("--bf16")

    print(f"[train] {' '.join(cmd)}")

    wandb_url = None
    wandb_pat = re.compile(r"https://wandb\.ai/\S+")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=repo_dir,
        env={**os.environ, "DATA_DIR": "/data/tokenized"},
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        if wandb_url is None:
            m = wandb_pat.search(line)
            if m:
                wandb_url = m.group(0)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"train.py exited {proc.returncode}")

    return {"wandb_url": wandb_url}


@app.function(
    image=image,
    volumes={"/data/tokenized": corpus_vol},
    timeout=60 * 60,
)
def upload_corpus():
    """Download corpus from HuggingFace into the Modal volume."""
    from huggingface_hub import hf_hub_download
    import shutil

    os.makedirs("/data/tokenized", exist_ok=True)
    for fname in ["corpus_train.bin", "corpus_val.bin"]:
        print(f"Downloading {fname} from HF ...")
        path = hf_hub_download(
            "tsuberim/merlin-corpus-v0",
            f"data/{fname}",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        dst = f"/data/tokenized/{fname}"
        shutil.copy(path, dst)
        size_gb = os.path.getsize(dst) / 1e9
        print(f"  {fname}: {size_gb:.2f} GB")

    corpus_vol.commit()
    print("Done. Corpus committed to volume.")


@app.local_entrypoint()
def main(
    commit: str = "",
    model: str = "experiment",
    batch_size: int = 256,
    max_steps: int = 20000,
    val_every: int = 500,
    val_steps: int = 10,
    save_every: int = 1000,
    bf16: bool = False,
):
    if not commit:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    # Validate commit is pushed
    branches = subprocess.run(
        ["git", "branch", "-r", "--contains", commit],
        capture_output=True, text=True,
    ).stdout.strip()
    if not branches:
        raise SystemExit(f"Commit {commit[:12]} not pushed to remote. Push first.")

    print(f"commit:  {commit[:12]}")
    print(f"model:   {model}  batch={batch_size}  steps={max_steps}\n")

    result = train.remote(
        commit=commit,
        model=model,
        batch_size=batch_size,
        max_steps=max_steps,
        val_every=val_every,
        val_steps=val_steps,
        save_every=save_every,
        bf16=bf16,
    )
    print(f"\nW&B: {result['wandb_url']}")
