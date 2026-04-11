"""
Modal app for merlin training on H100.

Deploy (needed once, or after requirements.txt changes):
    modal deploy modal_app.py

Run an experiment:
    modal run modal_app.py --commit <sha> [--model experiment] [--max-steps 200]

Build full corpus (download + filter + tokenize + upload to HF):
    modal run modal_app.py::build_corpus [--sources stack_python,stack_bash,...] [--full] [--workers 32]

Upload existing corpus to Modal volume (one-time):
    modal run modal_app.py::upload_corpus
"""
import os
import re
import subprocess
import sys

import modal

REPO_URL = "https://github.com/tsuberim/mllm.git"

app = modal.App("merlin-trainer")

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

# CPU-only image for data pipeline — lighter, cheaper, faster to build.
pipeline_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "datatrove[io]",
        "regex",
        "datasets",
        "huggingface_hub",
        "tokenizers",
        "tqdm",
        "numpy",
        "python-dotenv",
    )
)

# Single persistent volume: corpus/ cache/hf cache/inductor
vol = modal.Volume.from_name("merlin-data", create_if_missing=True)

DATA_ROOT      = "/data"
HF_HOME        = f"{DATA_ROOT}/cache/hf"
INDUCTOR_CACHE = f"{DATA_ROOT}/cache/inductor"


@app.function(
    image=image,
    gpu="H100",
    volumes={DATA_ROOT: vol},
    timeout=60 * 60 * 12,  # 12h max
    secrets=[modal.Secret.from_name("merlin")],
)
def train(
    commit: str,
    model: str = "experiment",
    batch_size: int = 128,
    max_steps: int = 20000,
    val_every: int = 2000,
    val_steps: int = 10,
    save_every: int = 1000,
    bf16: bool = True,
    grad_checkpoint: bool = False,
    lr_min: float = 0.0,
    warmup: int = 0,
    ema_count: float = 0.0,
    resume: bool = False,
    tag: str = "",
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
    if grad_checkpoint:
        cmd.append("--grad_checkpoint")
    if lr_min > 0:
        cmd += ["--lr_min", str(lr_min)]
    if warmup > 0:
        cmd += ["--warmup", str(warmup)]
    if ema_count > 0:
        cmd += ["--ema_count", str(ema_count)]
    if resume:
        cmd.append("--resume")
    cmd += ["--tag", tag or commit[:12]]

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
        env={
            **os.environ,
            "DATA_DIR":                f"{DATA_ROOT}/tokenized",
            "HF_HOME":                 HF_HOME,
            "TORCHINDUCTOR_CACHE_DIR": INDUCTOR_CACHE,
        },
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        if wandb_url is None:
            m = wandb_pat.search(line)
            if m:
                wandb_url = m.group(0)
                print(f"\n>>> W&B: {wandb_url}\n", flush=True)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"train.py exited {proc.returncode}")

    return {"wandb_url": wandb_url}


@app.function(
    image=pipeline_image,
    cpu=32,
    memory=32768,  # 32 GB — DataTrove streams, but filter stages need headroom
    volumes={DATA_ROOT: vol},
    timeout=60 * 60 * 12,  # 12h max
    secrets=[modal.Secret.from_name("merlin")],
)
def build_corpus(
    commit: str,
    sources: str = "stack_python,stack_bash,stack_md,stackoverflow,github_commits,github_issues",
    full: bool = False,
    workers: int = 28,
    hf_corpus_repo: str = "tsuberim/merlin-corpus-v1",
):
    """
    Full corpus pipeline on a CPU machine:
      1. pipeline.py  — DataTrove download + filter per source  → /data/processed/
      2. 05_tokenize.py — multiprocess tokenize + pack          → /data/tokenized-new/
      3. Upload corpus_train.bin + corpus_val.bin to HF dataset repo
    Resumable: re-run with same args to continue from last completed shard/source.
    """
    import shutil

    repo_dir = f"/tmp/{commit}"
    subprocess.run(
        ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", REPO_URL, repo_dir],
        check=True,
    )
    subprocess.run(
        ["git", "-C", repo_dir, "checkout", "--quiet", "--detach", commit],
        check=True,
    )
    print(f"[pipeline] checked out {commit[:12]}")

    processed_dir  = f"{DATA_ROOT}/processed"
    tokenized_dir  = f"{DATA_ROOT}/tokenized-new"
    logs_dir       = f"{DATA_ROOT}/pipeline-logs"
    tok_dir        = f"{repo_dir}/data/tokenizer"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(tokenized_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = {
        **os.environ,
        "HF_HOME": HF_HOME,
    }

    # ── step 1: DataTrove pipeline ─────────────────────────────────────────────
    print(f"\n[pipeline] step 1: download + filter (sources={sources}, full={full})")
    cmd1 = [
        sys.executable, "-u", f"{repo_dir}/data/pipeline/pipeline.py",
        "--out",     processed_dir,
        "--logs",    logs_dir,
        "--workers", str(workers),
    ] + (["--full"] if full else [])
    for source in sources.split(","):
        cmd1 += ["--source", source.strip()]

    proc = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=repo_dir, env=env)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"pipeline.py exited {proc.returncode}")
    vol.commit()

    # ── step 2: tokenize + pack ────────────────────────────────────────────────
    print(f"\n[pipeline] step 2: tokenize + pack")
    cmd2 = [
        sys.executable, "-u", f"{repo_dir}/data/pipeline/05_tokenize.py",
        "--in",      processed_dir,
        "--tok",     tok_dir,
        "--out",     tokenized_dir,
        "--workers", str(workers),
    ]
    proc = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=repo_dir, env=env)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"05_tokenize.py exited {proc.returncode}")
    vol.commit()

    # ── step 3: upload to HF ──────────────────────────────────────────────────
    print(f"\n[pipeline] step 3: uploading to {hf_corpus_repo}")
    from huggingface_hub import HfApi, create_repo
    hf = HfApi()
    create_repo(hf_corpus_repo, repo_type="dataset", exist_ok=True,
                token=os.environ.get("HF_TOKEN"))
    for fname in ["corpus_train.bin", "corpus_val.bin"]:
        src = f"{tokenized_dir}/{fname}"
        if not os.path.exists(src):
            print(f"  WARNING: {fname} not found, skipping")
            continue
        size_gb = os.path.getsize(src) / 1e9
        print(f"  uploading {fname} ({size_gb:.2f} GB) ...")
        hf.upload_file(
            path_or_fileobj=src,
            path_in_repo=f"data/{fname}",
            repo_id=hf_corpus_repo,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"    done")

    print(f"\n[pipeline] corpus published to {hf_corpus_repo}")
    return {"repo": hf_corpus_repo}


@app.function(
    image=image,
    volumes={DATA_ROOT: vol},
    timeout=60 * 60,
)
def upload_corpus():
    """Download corpus from HuggingFace into the Modal volume."""
    from huggingface_hub import hf_hub_download
    import shutil

    tokenized_dir = f"{DATA_ROOT}/tokenized"
    os.makedirs(tokenized_dir, exist_ok=True)
    for fname in ["corpus_train.bin", "corpus_val.bin"]:
        print(f"Downloading {fname} from HF ...")
        path = hf_hub_download(
            "tsuberim/merlin-corpus-v0",
            f"data/{fname}",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        dst = f"{tokenized_dir}/{fname}"
        shutil.copy(path, dst)
        size_gb = os.path.getsize(dst) / 1e9
        print(f"  {fname}: {size_gb:.2f} GB")

    vol.commit()
    print("Done. Corpus committed to volume.")


@app.local_entrypoint()
def build_corpus_entrypoint(
    commit: str = "",
    sources: str = "stack_python,stack_bash,stack_md,stackoverflow,github_commits,github_issues",
    full: bool = False,
    workers: int = 28,
    hf_corpus_repo: str = "tsuberim/merlin-corpus-v1",
):
    """
    Launch corpus pipeline on Modal CPU machine.
        modal run modal_app.py::build_corpus_entrypoint [--full] [--sources stack_python,...]
    """
    if not commit:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    branches = subprocess.run(
        ["git", "branch", "-r", "--contains", commit],
        capture_output=True, text=True,
    ).stdout.strip()
    if not branches:
        raise SystemExit(f"Commit {commit[:12]} not pushed to remote. Push first.")

    print(f"commit:  {commit[:12]}")
    print(f"sources: {sources}")
    print(f"full:    {full}  workers: {workers}\n")

    result = build_corpus.remote(
        commit=commit,
        sources=sources,
        full=full,
        workers=workers,
        hf_corpus_repo=hf_corpus_repo,
    )
    print(f"\nCorpus published: {result['repo']}")


@app.local_entrypoint()
def main(
    commit: str = "",
    model: str = "experiment",
    batch_size: int = 128,
    max_steps: int = 20000,
    val_every: int = 2000,
    val_steps: int = 5,
    save_every: int = 1000,
    bf16: bool = True,
    grad_checkpoint: bool = False,
    lr_min: float = 0.0,
    warmup: int = 0,
    ema_count: float = 0.0,
    resume: bool = False,
    tag: str = "",
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
        grad_checkpoint=grad_checkpoint,
        lr_min=lr_min,
        warmup=warmup,
        ema_count=ema_count,
        resume=resume,
        tag=tag,
    )
    print(f"\nW&B: {result['wandb_url']}")
