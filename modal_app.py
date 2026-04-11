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
    tok_dir = f"{repo_dir}/data/tokenizer"
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
