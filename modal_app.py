"""
Modal app for merlin training on H100.

Deploy (needed once, or after requirements.txt changes):
    modal deploy modal_app.py

Run pre-training experiment:
    modal run modal_app.py --commit <sha> [--model experiment] [--max-steps 200]

Prepare SFT data (CPU, one-time or when data changes):
    modal run modal_app.py::prepare_sft_entrypoint [--dataset HuggingFaceH4/CodeAlpaca_20K]

Run SFT fine-tuning:
    modal run modal_app.py::sft_entrypoint --commit <sha> [--model experiment] [--base experiment]

Build full corpus (download + filter + tokenize + upload to HF):
    modal run modal_app.py::build_corpus [--sources stack_python,stack_bash,...] [--full] [--workers 32]

Upload existing corpus to Modal volume (one-time):
    modal run modal_app.py::upload_corpus
"""
from __future__ import annotations

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


def _checkout(commit: str) -> str:
    """Clone repo at commit into /tmp/<commit>. Idempotent."""
    import shutil
    repo_dir = f"/tmp/{commit}"
    if not os.path.exists(f"{repo_dir}/.git"):
        # Remove stale/partial directory if present
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        result = subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, repo_dir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed:\n{result.stderr}")
    result = subprocess.run(
        ["git", "-C", repo_dir, "checkout", "--detach", commit],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git checkout failed:\n{result.stderr}")
    return repo_dir


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
    batch_size: int = 40,
    max_steps: int = 10000,
    val_every: int = 1000,
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
    memory=4096,   # DataTrove streams — RAM usage is O(batch), not O(corpus)
    volumes={DATA_ROOT: vol},
    timeout=60 * 60 * 6,  # 6h per source
    secrets=[modal.Secret.from_name("merlin")],
)
def filter_source(commit: str, source: str, full: bool = False, workers: int = 28) -> dict:
    """Download + filter a single source. One Modal container per source, run in parallel."""
    repo_dir = _checkout(commit)
    processed_dir = f"{DATA_ROOT}/processed"
    logs_dir      = f"{DATA_ROOT}/pipeline-logs"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = {**os.environ, "HF_HOME": HF_HOME}
    cmd = [
        sys.executable, "-u", f"{repo_dir}/data/pipeline/pipeline.py",
        "--source",  source,
        "--out",     processed_dir,
        "--logs",    logs_dir,
        "--workers", str(workers),
    ] + (["--full"] if full else [])

    import time
    print(f"[filter_source:{source}] starting")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            cwd=repo_dir, env=env)
    for line in proc.stdout:
        print(line.decode("utf-8", errors="replace"), end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"pipeline.py exited {proc.returncode} for source={source}")
    vol.commit()
    elapsed = time.time() - t0
    print(f"[filter_source:{source}] done  ({elapsed:.0f}s)")
    return {"source": source, "elapsed": elapsed}


@app.function(
    image=pipeline_image,
    cpu=32,
    memory=4096,   # Phase 1 is CPU-bound per shard, RAM usage is O(shard), not O(corpus)
    volumes={DATA_ROOT: vol},
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name("merlin")],
)
def tokenize_phase1(commit: str, source: str, workers: int = 28) -> dict:
    """Phase 1: tokenize shards for one source → .tok.bin + .idx.bin. Run in parallel."""
    repo_dir = _checkout(commit)
    vol.reload()

    processed_dir = f"{DATA_ROOT}/processed"
    tokenized_dir = f"{DATA_ROOT}/tokenized-new"
    tok_dir       = f"{DATA_ROOT}/tokenizer"
    os.makedirs(tokenized_dir, exist_ok=True)

    import time
    t0 = time.time()
    env = {**os.environ, "HF_HOME": HF_HOME}
    cmd = [
        sys.executable, "-u", f"{repo_dir}/data/pipeline/05_tokenize.py",
        "--in", processed_dir, "--tok", tok_dir, "--out", tokenized_dir,
        "--workers", str(workers), "--phase", "1", "--source", source,
    ]
    print(f"[tokenize_phase1:{source}] starting")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            cwd=repo_dir, env=env)
    for line in proc.stdout:
        print(line.decode("utf-8", errors="replace"), end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"05_tokenize.py phase1 exited {proc.returncode}")
    vol.commit()
    elapsed = time.time() - t0
    print(f"[tokenize_phase1:{source}] done  ({elapsed:.0f}s)")
    return {"source": source, "elapsed": elapsed}


@app.function(
    image=pipeline_image,
    cpu=4,
    memory=32768,  # Phase 2 loads full index into RAM: ~2.4GB/100B tokens; headroom for full corpus
    volumes={DATA_ROOT: vol},
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name("merlin")],
)
def tokenize_phase2(commit: str) -> dict:
    """Phase 2: global shuffle + stream-pack → corpus_train.bin / corpus_val.bin."""
    repo_dir = _checkout(commit)
    vol.reload()

    tokenized_dir = f"{DATA_ROOT}/tokenized-new"
    tok_dir       = f"{DATA_ROOT}/tokenizer"

    import time
    t0 = time.time()
    env = {**os.environ, "HF_HOME": HF_HOME}
    cmd = [
        sys.executable, "-u", f"{repo_dir}/data/pipeline/05_tokenize.py",
        "--in", f"{DATA_ROOT}/processed", "--tok", tok_dir, "--out", tokenized_dir,
        "--phase", "2",
    ]
    print("[tokenize_phase2] starting")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            cwd=repo_dir, env=env)
    for line in proc.stdout:
        print(line.decode("utf-8", errors="replace"), end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"05_tokenize.py phase2 exited {proc.returncode}")
    vol.commit()
    elapsed = time.time() - t0
    print(f"[tokenize_phase2] done  ({elapsed:.0f}s)")
    return {"tokenized_dir": tokenized_dir, "elapsed": elapsed}


@app.function(
    image=pipeline_image,
    cpu=2,
    memory=2048,
    volumes={DATA_ROOT: vol},
    timeout=60 * 60 * 12,
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
    Full corpus pipeline — all steps parallelized across sources where possible.
      Step 1: filter_source × N      (parallel, one container per source)
      Step 2a: tokenize_phase1 × N   (parallel, one container per source)
      Step 2b: tokenize_phase2       (single container — global shuffle + pack)
      Step 3: upload to HF
    Resumable: DataTrove tracks completed shards; phase1 skips existing .tok.bin files.
    """
    import time
    source_list = [s.strip() for s in sources.split(",")]
    print(f"[build_corpus] sources={source_list}  full={full}  workers={workers}")
    t_total = time.time()

    # ── step 1: parallel filter ────────────────────────────────────────────────
    print("\n[build_corpus] step 1: filter (parallel per source)")
    t0 = time.time()
    results = list(filter_source.starmap([(commit, src, full, workers) for src in source_list]))
    t1 = time.time()
    print(f"[build_corpus] step 1 done: {[r['source'] for r in results]}  ({t1-t0:.0f}s)")

    # ── step 2a: parallel tokenize phase 1 ────────────────────────────────────
    print("\n[build_corpus] step 2a: tokenize phase 1 (parallel per source)")
    t0 = time.time()
    p1_results = list(tokenize_phase1.starmap([(commit, src, workers) for src in source_list]))
    t1 = time.time()
    print(f"[build_corpus] step 2a done: {[r['source'] for r in p1_results]}  ({t1-t0:.0f}s)")

    # ── step 2b: tokenize phase 2 (serial global pack) ────────────────────────
    print("\n[build_corpus] step 2b: tokenize phase 2 (global pack)")
    t0 = time.time()
    tokenize_phase2.remote(commit)
    t1 = time.time()
    print(f"[build_corpus] step 2b done  ({t1-t0:.0f}s)")

    # ── step 3: upload to HF ──────────────────────────────────────────────────
    vol.reload()
    tokenized_dir = f"{DATA_ROOT}/tokenized-new"
    print(f"\n[build_corpus] step 3: uploading to {hf_corpus_repo}")
    t0 = time.time()
    from huggingface_hub import HfApi, create_repo
    hf = HfApi()
    create_repo(hf_corpus_repo, repo_type="dataset", exist_ok=True,
                token=os.environ.get("HF_TOKEN"))
    for fname in ["corpus_train.bin", "corpus_val.bin"]:
        src_path = f"{tokenized_dir}/{fname}"
        if not os.path.exists(src_path):
            print(f"  WARNING: {fname} not found, skipping")
            continue
        size_gb = os.path.getsize(src_path) / 1e9
        print(f"  uploading {fname} ({size_gb:.2f} GB) ...")
        hf.upload_file(
            path_or_fileobj=src_path,
            path_in_repo=f"data/{fname}",
            repo_id=hf_corpus_repo,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"    done")
    t1 = time.time()
    print(f"[build_corpus] step 3 done  ({t1-t0:.0f}s)")
    print(f"\n[build_corpus] corpus published to {hf_corpus_repo}  total={t1-t_total:.0f}s")
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


@app.function(
    image=pipeline_image,
    cpu=4,
    memory=8192,
    volumes={DATA_ROOT: vol},
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("merlin")],
)
def prepare_sft(
    commit: str,
    dataset: str = "HuggingFaceH4/CodeAlpaca_20K",
    seq_len: int = 2048,
    max_examples: int = 0,
):
    """Download + tokenize SFT data into the Modal volume (CPU-only, cheap)."""
    repo_dir = f"/tmp/{commit}"
    subprocess.run(
        ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", REPO_URL, repo_dir],
        check=True,
    )
    subprocess.run(
        ["git", "-C", repo_dir, "checkout", "--quiet", "--detach", commit],
        check=True,
    )
    print(f"[prepare_sft] checked out {commit[:12]}")

    sft_dir = f"{DATA_ROOT}/sft"
    tok_dir = f"{DATA_ROOT}/tokenizer"
    cmd = [
        sys.executable, "-u", f"{repo_dir}/data/pipeline/06_prepare_sft.py",
        "--tok",      tok_dir,
        "--out",      sft_dir,
        "--dataset",  dataset,
        "--seq-len",  str(seq_len),
    ]
    if max_examples:
        cmd += ["--max-examples", str(max_examples)]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=repo_dir,
        env={**os.environ, "HF_HOME": HF_HOME},
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"06_prepare_sft.py exited {proc.returncode}")

    vol.commit()
    print(f"[prepare_sft] done → {sft_dir}")


@app.function(
    image=image,
    gpu="H100",
    volumes={DATA_ROOT: vol},
    timeout=60 * 60 * 12,
    secrets=[modal.Secret.from_name("merlin")],
)
def sft(
    commit: str,
    model: str = "experiment",
    base: str = "experiment",
    max_steps: int = 1000,
    val_every: int = 100,
    val_steps: int = 5,
    save_every: int = 500,
    batch_size: int = 16,
    lr: float = 1e-5,
    bf16: bool = True,
    tag: str = "",
) -> dict:
    """SFT fine-tuning on H100. Run prepare_sft first to populate the volume."""
    repo_dir = f"/tmp/{commit}"
    subprocess.run(
        ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", REPO_URL, repo_dir],
        check=True,
    )
    subprocess.run(
        ["git", "-C", repo_dir, "checkout", "--quiet", "--detach", commit],
        check=True,
    )
    print(f"[sft] checked out {commit[:12]}")

    cmd = [
        sys.executable, "-u", f"{repo_dir}/sft.py",
        "--model",      model,
        "--base",       base,
        "--max_steps",  str(max_steps),
        "--val_every",  str(val_every),
        "--val_steps",  str(val_steps),
        "--save_every", str(save_every),
        "--batch_size", str(batch_size),
        "--lr",         str(lr),
    ]
    if bf16:
        cmd.append("--bf16")
    if tag:
        cmd += ["--tag", tag]

    print(f"[sft] {' '.join(cmd)}")

    import re as _re
    wandb_url = None
    wandb_pat = _re.compile(r"https://wandb\.ai/\S+")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=repo_dir,
        env={
            **os.environ,
            "SFT_DIR":                 f"{DATA_ROOT}/sft",
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
        raise RuntimeError(f"sft.py exited {proc.returncode}")

    return {"wandb_url": wandb_url}


@app.local_entrypoint()
def prepare_sft_entrypoint(
    commit: str = "",
    dataset: str = "HuggingFaceH4/CodeAlpaca_20K",
    seq_len: int = 2048,
    max_examples: int = 0,
):
    """
    Prepare SFT data on a CPU Modal machine.
        modal run modal_app.py::prepare_sft_entrypoint [--dataset ...]
    """
    if not commit:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branches = subprocess.run(
        ["git", "branch", "-r", "--contains", commit],
        capture_output=True, text=True,
    ).stdout.strip()
    if not branches:
        raise SystemExit(f"Commit {commit[:12]} not pushed to remote. Push first.")

    print(f"commit: {commit[:12]}  dataset: {dataset}")
    prepare_sft.remote(
        commit=commit, dataset=dataset, seq_len=seq_len,
        max_examples=max_examples,
    )
    print("SFT data ready in Modal volume.")


@app.local_entrypoint()
def sft_entrypoint(
    commit: str = "",
    model: str = "experiment",
    base: str = "experiment",
    max_steps: int = 1000,
    val_every: int = 100,
    val_steps: int = 5,
    save_every: int = 500,
    batch_size: int = 16,
    lr: float = 1e-5,
    bf16: bool = True,
    tag: str = "",
):
    """
    Run SFT on H100.
        modal run modal_app.py::sft_entrypoint --model experiment --base experiment
    """
    if not commit:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branches = subprocess.run(
        ["git", "branch", "-r", "--contains", commit],
        capture_output=True, text=True,
    ).stdout.strip()
    if not branches:
        raise SystemExit(f"Commit {commit[:12]} not pushed to remote. Push first.")

    print(f"commit: {commit[:12]}  model: {model}  base: {base}  steps: {max_steps}")
    result = sft.remote(
        commit=commit, model=model, base=base, max_steps=max_steps,
        val_every=val_every, val_steps=val_steps, save_every=save_every,
        batch_size=batch_size, lr=lr, bf16=bf16, tag=tag,
    )
    print(f"\nW&B: {result['wandb_url']}")


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


@app.function(
    image=pipeline_image,
    cpu=1,
    memory=2048,
    timeout=5 * 60,   # 5 min max per repo
    retries=0,        # bad repos stay bad
)
def scan_repo(repo_json: str) -> str | None:
    """
    Clone, validate, and pytest a single repo in an isolated Modal container.
    Returns JSON string (repo metadata + status) if ≥20 tests pass, else None.
    No Docker needed — the container IS the sandbox.
    """
    import json, re, shutil, subprocess, sys
    from pathlib import Path

    repo = json.loads(repo_json)
    full_name = repo["full_name"]
    repo_dir = Path(f"/tmp/{full_name.replace('/', '__')}")

    # ── clone ──────────────────────────────────────────────────────────────────
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", "--quiet",
             f"https://github.com/{full_name}.git", str(repo_dir)],
            check=True, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    try:
        # ── static service check ───────────────────────────────────────────────
        _SERVICE_PKGS = {
            "psycopg2", "psycopg2-binary", "psycopg", "asyncpg", "pymysql",
            "pymongo", "motor", "redis", "aioredis", "elasticsearch",
            "cassandra-driver", "neo4j", "influxdb", "influxdb-client",
            "boto3", "botocore", "google-cloud-storage", "google-cloud-bigquery",
            "azure-storage-blob", "pika", "confluent-kafka", "kafka-python",
            "celery", "testcontainers", "pytest-docker", "sendgrid", "twilio",
            "ldap3", "paramiko",
        }
        _SVC_RE = re.compile(
            r"psycopg2\.connect\(|asyncpg\.connect\(|pymysql\.connect\("
            r"|pymongo\.MongoClient\(|redis\.Redis\(|redis\.StrictRedis\("
            r"|testcontainers\.|DockerContainer\(|from docker import|import docker"
            r"|boto3\.client\(|boto3\.resource\(|storage\.Client\("
            r"|os\.environ\[.(?:DATABASE_URL|REDIS_URL|MONGO_URI|BROKER_URL)"
            r"|os\.getenv\(.(?:DATABASE_URL|REDIS_URL|MONGO_URI|BROKER_URL)",
            re.IGNORECASE,
        )
        _SQLITE_RE = re.compile(r'create_engine\(["\']sqlite', re.IGNORECASE)

        for req in list(repo_dir.glob("requirements*.txt")) + list(repo_dir.glob("*/requirements*.txt")):
            try:
                if any(pkg in req.read_text(errors="replace").lower() for pkg in _SERVICE_PKGS):
                    return None
            except Exception:
                pass
        pyproject = repo_dir / "pyproject.toml"
        if pyproject.exists():
            try:
                if any(pkg in pyproject.read_text(errors="replace").lower() for pkg in _SERVICE_PKGS):
                    return None
            except Exception:
                pass
        for tf in list(repo_dir.rglob("conftest.py")) + list(repo_dir.rglob("test_*.py")):
            try:
                src = tf.read_text(errors="replace")
                if _SVC_RE.search(src) and not _SQLITE_RE.search(src):
                    return None
            except Exception:
                pass

        # ── require enough test files ──────────────────────────────────────────
        n_test_files = sum(
            1 for p in repo_dir.rglob("*.py")
            if p.name.startswith("test_") or p.name.endswith("_test.py")
            or "/tests/" in str(p) or "/test/" in str(p)
        )
        if n_test_files < 5:
            return None

        # ── install deps ───────────────────────────────────────────────────────
        venv = repo_dir / ".venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=30)
        pip = str(venv / "bin" / "pip")
        py  = str(venv / "bin" / "python")

        subprocess.run(
            [pip, "install", "pytest", "-q", "--no-warn-script-location"],
            check=True, timeout=120, cwd=str(repo_dir),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [pip, "install", "-e", ".", "-q", "--no-warn-script-location"],
            timeout=120, cwd=str(repo_dir),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for req in repo_dir.glob("requirements*.txt"):
            subprocess.run(
                [pip, "install", "-r", str(req), "-q", "--no-warn-script-location"],
                timeout=120, cwd=str(repo_dir),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        # ── run pytest ─────────────────────────────────────────────────────────
        r = subprocess.run(
            [py, "-m", "pytest", "--tb=no", "-q", "--no-header",
             "--continue-on-collection-errors"],
            capture_output=True, text=True, timeout=90, cwd=str(repo_dir),
        )
        failing, n_passed = [], 0
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("FAILED "):
                failing.append(line[7:].split(" - ")[0])
            m = re.search(r"(\d+) passed", line)
            if m:
                n_passed = int(m.group(1))

        if r.returncode == 5 or (n_passed == 0 and not failing):
            return None
        if failing:
            return None
        if n_passed < 20:
            return None

        repo["pytest_status"] = f"PASS ({n_passed} tests)"
        repo["n_tests"] = n_passed
        return json.dumps(repo)

    finally:
        shutil.rmtree(repo_dir, ignore_errors=True)


@app.local_entrypoint()
def scan_repos(
    candidates: str = "data/repos/candidates.jsonl",
    out: str = "data/repos/passing.jsonl",
):
    """
    Scan all candidates in parallel on Modal CPU containers.

        modal run modal_app.py::scan_repos
        modal run modal_app.py::scan_repos --candidates path/to/candidates.jsonl
    """
    import json
    from pathlib import Path

    repos = []
    with open(candidates) as f:
        for line in f:
            try:
                repos.append(json.loads(line.strip()))
            except Exception:
                pass
    print(f"Scanning {len(repos)} repos on Modal (one container per repo)...")

    results = scan_repo.map(
        [json.dumps(r) for r in repos],
        return_exceptions=True,
    )

    passing, errors = [], 0
    for r in results:
        if isinstance(r, Exception):
            errors += 1
        elif r:
            passing.append(r)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in passing:
            f.write(r + "\n")

    total = len(repos)
    print(f"\nPassing: {len(passing)}/{total} ({100*len(passing)/max(total,1):.1f}%)")
    if errors:
        print(f"Errors:  {errors}")
    print(f"Written: {out}")


@app.function(
    image=pipeline_image,
    cpu=2,
    memory=4096,
    timeout=15 * 60,   # 15 min per repo
    retries=0,
    volumes={DATA_ROOT: vol},
)
def mutate_repo(repo_json: str, commit: str, n_tasks: int = 20) -> list[str]:
    """
    Clone a repo, install deps, generate AST-mutation tasks, zip the repo for later use.
    Runs on Modal CPU — uses LocalSandbox (no Docker needed inside the container).

    Writes one zip to /data/repos/zips/{owner}__{name}@{commit}.zip (idempotent).
    Returns task records as JSON strings; each record includes repo_zip_key for lookup.

    Schema: id, task_name, task_category, repo, repo_commit, repo_zip_key, repo_stars,
            repo_topics, repo_license, mutation_kind, mutation_description, mutated_file,
            mutation_lineno, mutation_patch, failing_tests, n_failing_tests, task, test_snapshots
    """
    import hashlib
    import json
    import shutil
    import sys
    from pathlib import Path

    repo = json.loads(repo_json)
    full_name = repo["full_name"]
    slug = full_name.replace("/", "__")
    repo_dir = Path(f"/tmp/repos/{slug}")

    # Clone mllm at commit to import harness (cached across warm containers)
    mllm_dir = Path("/tmp/mllm")
    if not (mllm_dir / "harness").exists():
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout",
             REPO_URL, str(mllm_dir)],
            check=True, capture_output=True,
        )
    subprocess.run(
        ["git", "-C", str(mllm_dir), "checkout", "--quiet", "--detach", commit],
        check=True, capture_output=True,
    )
    if str(mllm_dir) not in sys.path:
        sys.path.insert(0, str(mllm_dir))

    from harness.sandbox import LocalSandbox
    from harness.task_gen import generate_tasks

    # Clone target repo
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", "--quiet",
             f"https://github.com/{full_name}.git", str(repo_dir)],
            check=True, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    try:
        tasks = generate_tasks(
            repo_path=repo_dir,
            n_tasks=n_tasks,
            sandbox_class=LocalSandbox,
        )
        if not tasks:
            return []

        repo_commit = tasks[0].repo_commit
        zip_key = f"{slug}@{repo_commit}"

        # Zip the clean repo (pre-mutation) — used at trace gen time to reconstruct
        # the environment without re-cloning. Idempotent: skip if already zipped.
        zips_dir = Path(f"{DATA_ROOT}/repos/zips")
        zips_dir.mkdir(parents=True, exist_ok=True)
        zip_path = zips_dir / f"{zip_key}.zip"
        if not zip_path.exists():
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(repo_dir))
        vol.commit()

        records = []
        for task in tasks:
            id_str = f"{full_name}:{task.mutated_file}:{task.mutation.kind}:{task.mutation.lineno}"
            task_id = hashlib.sha1(id_str.encode()).hexdigest()[:16]
            record = {
                "id": task_id,
                "task_name": task.name,
                "task_category": task.category,
                "repo": full_name,
                "repo_commit": repo_commit,
                "repo_zip_key": zip_key,
                "repo_stars": repo.get("stars", 0),
                "repo_topics": repo.get("topics", []),
                "repo_license": repo.get("license", ""),
                "mutation_kind": task.mutation.kind,
                "mutation_description": task.mutation.description,
                "mutated_file": task.mutated_file,
                "mutation_lineno": task.mutation.lineno,
                "mutation_patch": task.mutation_patch,
                "failing_tests": task.failing_tests,
                "n_failing_tests": len(task.failing_tests),
                "task": task.instruction,
            }
            records.append(json.dumps(record, ensure_ascii=False))
        return records
    finally:
        shutil.rmtree(repo_dir, ignore_errors=True)


@app.local_entrypoint()
def generate_task_dataset(
    passing: str = "data/repos/passing.jsonl",
    out: str = "data/tasks/tasks.jsonl",
    n_tasks_per_repo: int = 20,
    commit: str = "",
):
    """
    Generate AST-mutation task dataset from passing repos on Modal CPU containers.

        modal run modal_app.py::generate_task_dataset
        modal run modal_app.py::generate_task_dataset --passing data/repos/passing.jsonl --n-tasks-per-repo 20
    """
    import json
    from pathlib import Path

    if not commit:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branches = subprocess.run(
        ["git", "branch", "-r", "--contains", commit],
        capture_output=True, text=True,
    ).stdout.strip()
    if not branches:
        raise SystemExit(f"Commit {commit[:12]} not pushed. Push first.")

    repos = []
    with open(passing) as f:
        for line in f:
            try:
                repos.append(json.loads(line.strip()))
            except Exception:
                pass
    print(f"Generating tasks for {len(repos)} repos × {n_tasks_per_repo} mutations on Modal...")

    results = mutate_repo.starmap(
        [(json.dumps(r), commit, n_tasks_per_repo) for r in repos],
        return_exceptions=True,
    )

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n_tasks, n_repos_ok, errors = 0, 0, 0
    with open(out, "w") as f:
        for batch in results:
            if isinstance(batch, Exception):
                errors += 1
                continue
            if batch:
                n_repos_ok += 1
                for record in batch:
                    f.write(record + "\n")
                    n_tasks += 1

    print(f"\nRepos processed: {n_repos_ok}/{len(repos)}  errors: {errors}")
    print(f"Tasks generated: {n_tasks}")
    print(f"Written: {out}")


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
