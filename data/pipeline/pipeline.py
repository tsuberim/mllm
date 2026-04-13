"""
Corpus pipeline: download + format per source, no filtering/dedup.
Sources match docs/dataset.md v1 target (experiment-scale caps applied).

Usage:
    python data/pipeline/pipeline.py --source stack_v2_python --out data/processed/
    python data/pipeline/pipeline.py --all --out data/processed/
    python data/pipeline/pipeline.py --all --full --workers 16 --out data/processed/
"""

import argparse
import gzip
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

MAX_FILE_BYTES = 1_000_000

# ── experiment caps (document count) ─────────────────────────────────────────
# Full-scale token budgets from docs/dataset.md; experiment caps are ~1% of full.

EXPERIMENT_CAPS = {
    # Code — Stack v2
    "stack_v2_python":      500_000,
    "stack_v2_ts":          100_000,
    "stack_v2_go":          100_000,
    "stack_v2_rust":         80_000,
    "stack_v2_bash":         80_000,
    "stack_v2_yaml":         50_000,
    "stack_v2_dockerfile":   30_000,
    "stack_v2_sql":          80_000,
    "stack_v2_md":          150_000,
    # Code — other
    "jupyter":               50_000,
    "rosetta_code":            None,   # tiny ~1K docs — always full
    # Q&A
    "stackoverflow":        200_000,
    "stack_exchange_other": 100_000,
    # Commits / issues
    "github_commits":       150_000,
    "github_issues":        100_000,
    # Reference — HF
    "wikibooks":             30_000,
    # Reference — direct download (small, always full)
    "tldr_pages":              None,
    "man_pages":               None,
    "python_docs":             None,
    "peps":                    None,
    "rfcs":                    None,
    # NL / general knowledge
    "fineweb_edu":          200_000,
    "arxiv":                 50_000,
    "wikipedia":             50_000,
    # Instruction following
    "flan_v2":              100_000,
    "natural_instructions":  50_000,
    "openhermes":            50_000,
    "nl2bash":                 None,   # ~9.3K pairs — always full
    # Math
    "numinamath":            50_000,
    "competition_math":      30_000,
    "proof_pile":            50_000,
    # Pedagogical — direct download
    "papers_with_code":      50_000,
    "pypi_readmes":          50_000,
    "fastai_notebooks":        None,
    "python_ds_handbook":      None,
    "sicp":                    None,
}

# ── Stack v2 ──────────────────────────────────────────────────────────────────
# bigcode/the-stack-v2-dedup: license-filtered, near-deduped by BigCode.

STACK_V2_LANGS = {
    "stack_v2_python":     "Python",
    "stack_v2_ts":         "TypeScript",
    "stack_v2_go":         "Go",
    "stack_v2_rust":       "Rust",
    "stack_v2_bash":       "Shell",
    "stack_v2_yaml":       "YAML",
    "stack_v2_dockerfile": "Dockerfile",
    "stack_v2_sql":        "SQL",
    "stack_v2_md":         "Markdown",
}

def _stack_v2_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    content = data.get("content") or ""
    if not content:
        return None
    return {
        "text": content[:MAX_FILE_BYTES],
        "id":   data.get("hexsha") or f"{path}/{id_in_file}",
        "metadata": {},
    }

def _stack_v2_pipeline(source: str, out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    lang  = STACK_V2_LANGS[source]
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS[source])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="bigcode/the-stack-v2-dedup",
            dataset_options={"name": lang, "split": "train"},
            adapter=_stack_v2_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / source)),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / source))


# ── Jupyter notebooks ─────────────────────────────────────────────────────────
# bigcode/starcoderdata: includes a deduplicated Jupyter scripts subset.

def _jupyter_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    content = data.get("content") or ""
    if not content:
        return None
    return {
        "text": content[:MAX_FILE_BYTES],
        "id":   str(id_in_file),
        "metadata": {},
    }

def _jupyter_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["jupyter"])
    limit = cap if cap is not None else -1
    # bigcode/starcoderdata has a single 'default' config containing all languages.
    # codeparrot/github-jupyter-parsed is a dedicated Jupyter dataset with a 'content' field.
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="codeparrot/github-jupyter-parsed",
            dataset_options={"split": "train"},
            adapter=_jupyter_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "jupyter")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "jupyter"))


# ── Rosetta Code ──────────────────────────────────────────────────────────────

def _rosetta_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    task = (data.get("task") or "").strip()
    lang = (data.get("language") or "").strip()
    code = (data.get("code") or "").strip()
    if not code:
        return None
    header = f"# {task}" if task else ""
    fence  = f"```{lang.lower()}\n{code}\n```" if lang else code
    return {
        "text": f"{header}\n\n{fence}".strip(),
        "id":   str(id_in_file),
        "metadata": {},
    }

def _rosetta_code_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["rosetta_code"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="cakiki/rosetta-code",
            dataset_options={"split": "train"},
            adapter=_rosetta_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "rosetta_code")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "rosetta_code"))


# ── stack overflow ────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENT_RE = re.compile(r"&(amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);")
_ENTITY_MAP  = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "nbsp": " "}

def _strip_html(text: str) -> str:
    text = _HTML_TAG_RE.sub("", text)
    def _ent(m):
        name = m.group(1)
        if name in _ENTITY_MAP: return _ENTITY_MAP[name]
        if name.startswith("#x"): return chr(int(name[2:], 16))
        if name.startswith("#"):  return chr(int(name[1:]))
        return " "
    return _HTML_ENT_RE.sub(_ent, text).strip()

def _so_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    answers = data.get("answers") or []
    if not answers:
        return None
    best = max(answers, key=lambda a: (a.get("selected", False), a.get("pm_score", 0)))
    answer = _strip_html(best.get("text", "")).strip()
    if not answer:
        return None
    question = _strip_html(data.get("question") or "")
    return {
        "text": f"Q: {question}\n\nA: {answer}",
        "id":   str(id_in_file),
        "metadata": {},
    }

def _stackoverflow_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["stackoverflow"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="HuggingFaceH4/stack-exchange-preferences",
            dataset_options={"split": "train"},
            adapter=_so_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "stackoverflow")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "stackoverflow"))


# ── Stack Exchange (other sites) ───────────────────────────────────────────────
# ArmelR/stack-exchange-instruction: Code Review, Unix, ServerFault, AskUbuntu,
# SoftwareEngineering, DevOps, DataScience, etc.

def _se_other_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    instruction = (data.get("instruction") or data.get("question") or "").strip()
    response    = (data.get("response") or data.get("answer") or "").strip()
    if not response:
        return None
    text = f"Q: {instruction}\n\nA: {response}" if instruction else response
    return {
        "text": text[:MAX_FILE_BYTES],
        "id":   str(id_in_file),
        "metadata": {},
    }

def _stack_exchange_other_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["stack_exchange_other"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="ArmelR/stack-exchange-instruction",
            dataset_options={"split": "train"},
            adapter=_se_other_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "stack_exchange_other")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "stack_exchange_other"))


# ── github commits ────────────────────────────────────────────────────────────

def _commits_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    subject = (data.get("subject") or "").strip()
    content = (data.get("new_contents") or "").strip()
    if not content:
        return None
    text = f"# {subject}\n\n{content}" if subject else content
    return {
        "text": text[:MAX_FILE_BYTES],
        "id":   str(id_in_file),
        "metadata": {},
    }

def _github_commits_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["github_commits"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="json",
            dataset_options={
                "data_files": {
                    "train": [
                        "hf://datasets/bigcode/commitpackft/data/python/data.jsonl",
                        "hf://datasets/bigcode/commitpackft/data/shell/data.jsonl",
                    ]
                },
                "split": "train",
            },
            adapter=_commits_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "github_commits")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "github_commits"))


# ── github issues ─────────────────────────────────────────────────────────────

_ISSUE_TOKENS = re.compile(r"<issue_start>|<issue_comment>|<issue_closed>|<issue_opened>")

def _issues_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    if data.get("pull_request"):
        return None
    content = (data.get("content") or "").strip()
    if not content:
        return None
    return {
        "text": _ISSUE_TOKENS.sub("\n---\n", content).strip()[:MAX_FILE_BYTES],
        "id":   str(id_in_file),
        "metadata": {},
    }

def _github_issues_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["github_issues"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="bigcode/the-stack-github-issues",
            dataset_options={"split": "train"},
            adapter=_issues_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "github_issues")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "github_issues"))


# ── Wikibooks ─────────────────────────────────────────────────────────────────
# wikimedia/wikibooks: English Wikibooks (2023-11-01 snapshot).

def _wikibooks_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    text  = (data.get("text") or "").strip()
    if not text:
        return None
    title = (data.get("title") or "").strip()
    body  = f"# {title}\n\n{text}" if title else text
    return {
        "text": body[:MAX_FILE_BYTES],
        "id":   data.get("id") or str(id_in_file),
        "metadata": {},
    }

def _wikibooks_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["wikibooks"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="wikimedia/wikibooks",
            dataset_options={"name": "20231101.en", "split": "train"},
            adapter=_wikibooks_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "wikibooks")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "wikibooks"))


# ── tldr-pages ────────────────────────────────────────────────────────────────

def _tldr_pages_pipeline(out_dir: Path, logs: Path):
    out_path  = out_dir / "tldr_pages"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("tldr_pages: already done, skipping")
        return

    url = "https://github.com/tldr-pages/tldr/archive/refs/heads/main.zip"
    try:
        print(f"tldr_pages: downloading {url} ...")
        with urllib.request.urlopen(url, timeout=60) as resp:
            content = resp.read()
    except Exception as e:
        print(f"tldr_pages: download failed: {e} — skipping")
        done_flag.touch()
        return

    out_file = out_path / "tldr_pages.jsonl.gz"
    n = 0
    with gzip.open(out_file, "wt") as f:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                if not name.endswith(".md"):
                    continue
                text = zf.read(name).decode("utf-8", errors="replace").strip()
                if text:
                    f.write(json.dumps({"text": text, "id": name}) + "\n")
                    n += 1
    done_flag.touch()
    print(f"tldr_pages: wrote {n} pages to {out_file}")


# ── Man pages ─────────────────────────────────────────────────────────────────
# linux/man-pages GitHub mirror: section 1 (commands), 2 (syscalls), 3 (library).
# Files are in troff/mdoc; we ship raw troff — tokenizer will see the markup.

def _man_pages_pipeline(out_dir: Path, logs: Path):
    out_path  = out_dir / "man_pages"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("man_pages: already done, skipping")
        return

    url = "https://github.com/mkerrisk/man-pages/archive/refs/heads/master.zip"
    try:
        print(f"man_pages: downloading {url} ...")
        with urllib.request.urlopen(url, timeout=120) as resp:
            content = resp.read()
    except Exception as e:
        print(f"man_pages: download failed: {e} — skipping")
        done_flag.touch()
        return

    out_file = out_path / "man_pages.jsonl.gz"
    n = 0
    with gzip.open(out_file, "wt") as f:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                # sections 1 (commands), 2 (syscalls), 3 (library), 7 (overview)
                parts = name.split("/")
                if len(parts) < 2:
                    continue
                section = parts[-2]  # e.g. "man1", "man2"
                if not re.match(r"^man[1237]$", section):
                    continue
                text = zf.read(name).decode("utf-8", errors="replace").strip()
                if text:
                    f.write(json.dumps({"text": text, "id": name}) + "\n")
                    n += 1
    done_flag.touch()
    print(f"man_pages: wrote {n} pages to {out_file}")


# ── Python docs ───────────────────────────────────────────────────────────────
# python.org hosts pre-built plaintext archives of all Python docs.

def _python_docs_pipeline(out_dir: Path, logs: Path):
    import tarfile, html
    out_path  = out_dir / "python_docs"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("python_docs: already done, skipping")
        return

    url = "https://docs.python.org/3/archives/python-3.13-docs-text.tar.bz2"
    try:
        print(f"python_docs: downloading {url} ...")
        with urllib.request.urlopen(url, timeout=120) as resp:
            content = resp.read()
    except Exception as e:
        print(f"python_docs: download failed: {e} — skipping")
        done_flag.touch()
        return

    out_file = out_path / "python_docs.jsonl.gz"
    n = 0
    with gzip.open(out_file, "wt") as f:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:bz2") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".txt"):
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                text = fobj.read().decode("utf-8", errors="replace").strip()
                if text:
                    doc_id = Path(member.name).stem
                    f.write(json.dumps({"text": text[:MAX_FILE_BYTES], "id": doc_id}) + "\n")
                    n += 1
    done_flag.touch()
    print(f"python_docs: wrote {n} docs to {out_file}")


# ── PEPs ──────────────────────────────────────────────────────────────────────
# python/peps GitHub repo: RST source for all Python Enhancement Proposals.

def _peps_pipeline(out_dir: Path, logs: Path):
    out_path  = out_dir / "peps"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("peps: already done, skipping")
        return

    url = "https://github.com/python/peps/archive/refs/heads/main.zip"
    try:
        print(f"peps: downloading {url} ...")
        with urllib.request.urlopen(url, timeout=60) as resp:
            content = resp.read()
    except Exception as e:
        print(f"peps: download failed: {e} — skipping")
        done_flag.touch()
        return

    out_file = out_path / "peps.jsonl.gz"
    _pep_re = re.compile(r"pep-\d{4}\.rst$")
    n = 0
    with gzip.open(out_file, "wt") as f:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                if not _pep_re.search(name):
                    continue
                text = zf.read(name).decode("utf-8", errors="replace").strip()
                if text:
                    doc_id = Path(name).stem
                    f.write(json.dumps({"text": text, "id": doc_id}) + "\n")
                    n += 1
    done_flag.touch()
    print(f"peps: wrote {n} PEPs to {out_file}")


# ── RFCs ──────────────────────────────────────────────────────────────────────
# IETF RFC editor provides full text RFCs. We fetch the ones most relevant to
# the dataset.md spec: HTTP, TLS, JSON, MIME, DNS, SMTP, TCP, IP, WebSocket.

_KEY_RFCS = [
    # HTTP
    7230, 7231, 7232, 7233, 7234, 7235,  # HTTP/1.1 suite
    7540,  # HTTP/2
    9110, 9111, 9112,  # HTTP semantics (latest)
    # TLS
    8446,  # TLS 1.3
    # JSON
    8259,  # JSON
    7159,  # JSON (superseded)
    6901,  # JSON Pointer
    6902,  # JSON Patch
    # DNS
    1034, 1035,
    # SMTP
    5321,
    # TCP / IP
    793, 791,
    # WebSocket
    6455,
    # OAuth / JWT
    6749, 7519, 7517,
    # MIME
    2045, 2046, 2047, 2048, 2049,
    # URI
    3986,
    # SSH
    4251, 4252, 4253,
    # Misc networking
    2616,  # HTTP/1.1 original (widely referenced)
    2119,  # RFC key words (MUST, SHOULD...)
]

def _rfcs_pipeline(out_dir: Path, logs: Path):
    out_path  = out_dir / "rfcs"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("rfcs: already done, skipping")
        return

    out_file = out_path / "rfcs.jsonl.gz"
    n = 0
    with gzip.open(out_file, "wt") as f:
        for rfc_num in _KEY_RFCS:
            url = f"https://www.rfc-editor.org/rfc/rfc{rfc_num}.txt"
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    text = resp.read().decode("utf-8", errors="replace").strip()
                if text:
                    f.write(json.dumps({"text": text[:MAX_FILE_BYTES], "id": f"rfc{rfc_num}"}) + "\n")
                    n += 1
                    print(f"  rfc{rfc_num}: ok")
            except Exception as e:
                print(f"  rfc{rfc_num}: failed ({e})")
    done_flag.touch()
    print(f"rfcs: wrote {n} RFCs to {out_file}")


# ── FineWeb-Edu ───────────────────────────────────────────────────────────────

def _fineweb_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    text = (data.get("text") or "").strip()
    if not text:
        return None
    return {
        "text": text[:MAX_FILE_BYTES],
        "id":   data.get("id") or str(id_in_file),
        "metadata": {},
    }

def _fineweb_edu_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["fineweb_edu"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="HuggingFaceFW/fineweb-edu",
            dataset_options={"name": "sample-10BT", "split": "train"},
            adapter=_fineweb_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "fineweb_edu")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "fineweb_edu"))


# ── ArXiv CS ──────────────────────────────────────────────────────────────────

def _arxiv_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    abstract = (data.get("abstract") or "").strip()
    article  = (data.get("article") or "").strip()
    if not article and not abstract:
        return None
    text = f"{abstract}\n\n{article}".strip() if abstract else article
    return {
        "text": text[:MAX_FILE_BYTES],
        "id":   str(id_in_file),
        "metadata": {},
    }

def _arxiv_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["arxiv"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="scientific_papers",
            dataset_options={"name": "arxiv", "split": "train", "trust_remote_code": True},
            adapter=_arxiv_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "arxiv")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "arxiv"))


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def _wikipedia_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    text = (data.get("text") or "").strip()
    if not text:
        return None
    title = (data.get("title") or "").strip()
    body  = f"# {title}\n\n{text}" if title else text
    return {
        "text": body[:MAX_FILE_BYTES],
        "id":   data.get("id") or str(id_in_file),
        "metadata": {},
    }

def _wikipedia_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["wikipedia"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="wikimedia/wikipedia",
            dataset_options={"name": "20231101.en", "split": "train"},
            adapter=_wikipedia_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "wikipedia")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "wikipedia"))


# ── FLAN v2 ───────────────────────────────────────────────────────────────────

def _flan_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    inp  = (data.get("inputs") or "").strip()
    out  = (data.get("targets") or "").strip()
    if not out:
        return None
    text = f"{inp}\n{out}" if inp else out
    return {
        "text": text[:MAX_FILE_BYTES],
        "id":   str(id_in_file),
        "metadata": {},
    }

def _flan_v2_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["flan_v2"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="Muennighoff/flan",
            dataset_options={"split": "train"},
            adapter=_flan_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "flan_v2")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "flan_v2"))


# ── Natural Instructions v2 ───────────────────────────────────────────────────

def _natural_instructions_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    inp = (data.get("input") or data.get("Instance", {}).get("input") or "").strip()
    out = (data.get("output") or "")
    if isinstance(out, list):
        out = out[0] if out else ""
    out = out.strip()
    if not out:
        return None
    definition = (data.get("Definition") or data.get("definition") or "")
    if isinstance(definition, list):
        definition = definition[0] if definition else ""
    definition = definition.strip()
    parts = []
    if definition:
        parts.append(definition)
    if inp:
        parts.append(f"Input: {inp}")
    parts.append(f"Output: {out}")
    return {
        "text": "\n\n".join(parts)[:MAX_FILE_BYTES],
        "id":   data.get("id") or str(id_in_file),
        "metadata": {},
    }

def _natural_instructions_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["natural_instructions"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="Muennighoff/natural-instructions",
            dataset_options={"split": "train"},
            adapter=_natural_instructions_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "natural_instructions")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "natural_instructions"))


# ── OpenHermes 2.5 ────────────────────────────────────────────────────────────

def _openhermes_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    convs = data.get("conversations") or []
    if not convs:
        return None
    parts = []
    for turn in convs:
        role  = (turn.get("from") or "").strip()
        value = (turn.get("value") or "").strip()
        if not value:
            continue
        label = "Human" if role == "human" else "Assistant"
        parts.append(f"{label}: {value}")
    if not parts:
        return None
    return {
        "text": "\n\n".join(parts)[:MAX_FILE_BYTES],
        "id":   str(id_in_file),
        "metadata": {},
    }

def _openhermes_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["openhermes"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="teknium/OpenHermes-2.5",
            dataset_options={"split": "train"},
            adapter=_openhermes_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "openhermes")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "openhermes"))


# ── NuminaMath ────────────────────────────────────────────────────────────────

def _numinamath_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    problem  = (data.get("problem") or "").strip()
    solution = (data.get("solution") or "").strip()
    if not problem or not solution:
        return None
    return {
        "text": f"Problem: {problem}\n\nSolution: {solution}",
        "id":   str(id_in_file),
        "metadata": {},
    }

def _numinamath_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["numinamath"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="AI-MO/NuminaMath-CoT",
            dataset_options={"split": "train"},
            adapter=_numinamath_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "numinamath")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "numinamath"))


# ── Competition / DeepMind Math ───────────────────────────────────────────────
# lighteval/MATH: Hendrycks MATH benchmark — competition problems with solutions.

def _competition_math_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    problem  = (data.get("problem") or "").strip()
    solution = (data.get("solution") or "").strip()
    if not problem or not solution:
        return None
    level   = data.get("level") or ""
    subject = data.get("type") or ""
    header  = f"[{subject} | {level}]\n" if subject or level else ""
    return {
        "text": f"{header}Problem: {problem}\n\nSolution: {solution}",
        "id":   str(id_in_file),
        "metadata": {},
    }

def _competition_math_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["competition_math"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="lighteval/MATH",
            dataset_options={"name": "all", "split": "train"},
            adapter=_competition_math_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "competition_math")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "competition_math"))


# ── Proof-Pile 2 ──────────────────────────────────────────────────────────────
# EleutherAI/proof-pile-2: 55B-token math corpus; taking arxiv-math subset.

def _proof_pile_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    text = (data.get("text") or "").strip()
    if not text:
        return None
    return {
        "text": text[:MAX_FILE_BYTES],
        "id":   str(id_in_file),
        "metadata": {},
    }

def _proof_pile_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["proof_pile"])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="EleutherAI/proof-pile-2",
            dataset_options={"name": "arxiv", "split": "train"},
            adapter=_proof_pile_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "proof_pile")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "proof_pile"))


# ── Papers with Code ──────────────────────────────────────────────────────────
# paperswithcode.com publishes a JSON dump of all paper abstracts.

def _papers_with_code_pipeline(out_dir: Path, logs: Path, limit=None):
    out_path  = out_dir / "papers_with_code"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("papers_with_code: already done, skipping")
        return

    url = "https://production-media.paperswithcode.com/about/papers-with-abstracts.json.gz"
    try:
        print(f"papers_with_code: downloading {url} ...")
        with urllib.request.urlopen(url, timeout=120) as resp:
            raw = resp.read()
    except Exception as e:
        print(f"papers_with_code: download failed: {e} — skipping")
        done_flag.touch()
        return

    try:
        papers = json.loads(gzip.decompress(raw))
    except Exception as e:
        print(f"papers_with_code: parse failed: {e} — skipping")
        done_flag.touch()
        return

    cap = limit if limit is not None else EXPERIMENT_CAPS.get("papers_with_code")
    out_file = out_path / "papers_with_code.jsonl.gz"
    n = 0
    with gzip.open(out_file, "wt") as f:
        for paper in papers:
            if cap and n >= cap:
                break
            title    = (paper.get("title") or "").strip()
            abstract = (paper.get("abstract") or "").strip()
            if not abstract:
                continue
            text = f"# {title}\n\n{abstract}" if title else abstract
            f.write(json.dumps({"text": text, "id": paper.get("paper_id") or str(n)}) + "\n")
            n += 1
    done_flag.touch()
    print(f"papers_with_code: wrote {n} papers to {out_file}")


# ── PyPI READMEs ──────────────────────────────────────────────────────────────
# Fetch package descriptions from the PyPI JSON API.
# Uses the top-pypi-packages list (30-day download stats) from GitHub.

def _pypi_readmes_pipeline(out_dir: Path, logs: Path, limit=None):
    out_path  = out_dir / "pypi_readmes"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("pypi_readmes: already done, skipping")
        return

    cap = limit if limit is not None else EXPERIMENT_CAPS.get("pypi_readmes") or 50_000

    # Top PyPI packages by downloads (public dataset from hugovk/top-pypi-packages)
    top_url = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"
    try:
        print(f"pypi_readmes: fetching package list from {top_url} ...")
        with urllib.request.urlopen(top_url, timeout=30) as resp:
            top_data = json.loads(resp.read())
        pkg_names = [r["project"] for r in top_data.get("rows", [])[:cap]]
    except Exception as e:
        print(f"pypi_readmes: failed to fetch package list: {e} — skipping")
        done_flag.touch()
        return

    out_file = out_path / "pypi_readmes.jsonl.gz"
    n = 0
    with gzip.open(out_file, "wt") as f:
        for pkg in pkg_names:
            url = f"https://pypi.org/pypi/{pkg}/json"
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    data = json.loads(resp.read())
                info = data.get("info") or {}
                description = (info.get("description") or "").strip()
                if not description or description == "UNKNOWN":
                    continue
                name    = info.get("name") or pkg
                version = info.get("version") or ""
                header  = f"# {name} {version}\n\n" if version else f"# {name}\n\n"
                f.write(json.dumps({
                    "text": (header + description)[:MAX_FILE_BYTES],
                    "id":   name,
                }) + "\n")
                n += 1
            except Exception:
                pass  # skip packages with no description or API errors
    done_flag.touch()
    print(f"pypi_readmes: wrote {n} package READMEs to {out_file}")


# ── GitHub-hosted pedagogical sources ─────────────────────────────────────────
# Small books / notebook collections fetched from GitHub as zip archives.
# Each is a direct-download source like nl2bash.

def _github_zip_notebooks(out_dir: Path, source_name: str, zip_url: str, extensions=(".ipynb",), timeout=120):
    """Generic helper: download a GitHub repo zip, extract notebook/text files."""
    out_path  = out_dir / source_name
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print(f"{source_name}: already done, skipping")
        return

    try:
        print(f"{source_name}: downloading {zip_url} ...")
        with urllib.request.urlopen(zip_url, timeout=timeout) as resp:
            content = resp.read()
    except Exception as e:
        print(f"{source_name}: download failed: {e} — skipping")
        done_flag.touch()
        return

    out_file = out_path / f"{source_name}.jsonl.gz"
    n = 0
    with gzip.open(out_file, "wt") as f:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                if not any(name.endswith(ext) for ext in extensions):
                    continue
                raw = zf.read(name)
                # Jupyter notebooks: extract cell sources as text
                if name.endswith(".ipynb"):
                    try:
                        nb   = json.loads(raw.decode("utf-8", errors="replace"))
                        cells = nb.get("cells") or nb.get("worksheets", [{}])[0].get("cells", [])
                        parts = []
                        for cell in cells:
                            src = cell.get("source") or cell.get("input") or ""
                            if isinstance(src, list):
                                src = "".join(src)
                            src = src.strip()
                            if src:
                                parts.append(src)
                        text = "\n\n".join(parts).strip()
                    except Exception:
                        text = raw.decode("utf-8", errors="replace").strip()
                else:
                    text = raw.decode("utf-8", errors="replace").strip()
                if text:
                    f.write(json.dumps({"text": text[:MAX_FILE_BYTES], "id": name}) + "\n")
                    n += 1
    done_flag.touch()
    print(f"{source_name}: wrote {n} docs to {out_file}")


def _fastai_notebooks_pipeline(out_dir: Path, logs: Path):
    _github_zip_notebooks(
        out_dir, "fastai_notebooks",
        "https://github.com/fastai/fastbook/archive/refs/heads/master.zip",
    )

def _python_ds_handbook_pipeline(out_dir: Path, logs: Path):
    _github_zip_notebooks(
        out_dir, "python_ds_handbook",
        "https://github.com/jakevdp/PythonDataScienceHandbook/archive/refs/heads/master.zip",
    )

def _sicp_pipeline(out_dir: Path, logs: Path):
    # SICP full text as HTML from the community mirror
    _github_zip_notebooks(
        out_dir, "sicp",
        "https://github.com/sarabander/sicp/archive/refs/heads/master.zip",
        extensions=(".html", ".xml"),
    )


# ── nl2bash ───────────────────────────────────────────────────────────────────

def _nl2bash_pipeline(out_dir: Path, logs: Path):
    out_path  = out_dir / "nl2bash"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("nl2bash: already done, skipping")
        return

    urls = [
        "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash_scripts.json",
        "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/all_invocations.json",
        "https://huggingface.co/datasets/jiacheng-ye/nl2bash/resolve/main/data/train-00000-of-00001.parquet",
    ]
    records = []
    for url in urls:
        try:
            print(f"nl2bash: trying {url} ...")
            with urllib.request.urlopen(url, timeout=30) as resp:
                content = resp.read()
            if url.endswith(".parquet"):
                import pyarrow.parquet as pq
                table = pq.read_table(io.BytesIO(content))
                df    = table.to_pydict()
                records = [{"nl": n, "bash": b} for n, b in zip(df.get("nl", []), df.get("bash", []))]
            else:
                raw = json.loads(content)
                if isinstance(raw, list):
                    records = raw
                elif isinstance(raw, dict):
                    for v in raw.values():
                        if isinstance(v, dict):
                            records.append({"nl": v.get("cmd_str", ""), "bash": v.get("invocation", "")})
                        elif isinstance(v, list):
                            records.extend(v)
            if records:
                break
        except Exception as e:
            print(f"nl2bash: {url} failed: {e}")

    if not records:
        print("nl2bash: WARNING — all fetch methods failed, skipping")
        done_flag.touch()
        return

    out_file = out_path / "nl2bash.jsonl.gz"
    n = 0
    with gzip.open(out_file, "wt") as f:
        for r in records:
            nl  = (r.get("nl") or r.get("cmd_str") or "").strip()
            cmd = (r.get("bash") or r.get("invocation") or "").strip()
            if nl and cmd:
                f.write(json.dumps({"text": f"# {nl}\n{cmd}", "id": nl[:40]}) + "\n")
                n += 1
    done_flag.touch()
    print(f"nl2bash: wrote {n} pairs to {out_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

# Not implemented (no public dataset / requires live scraping):
#   - Dev.to + HashNode (2B): FineWeb-Edu sample-10BT likely covers much of this
#   - Git book + Docker docs + Bash manual (0.5B): scraping required
#   - Library docs NumPy/Pandas/etc (0.1B): ReadTheDocs scraping required
#   - Math Jupyter notebooks (0.5B): largely covered by `jupyter` source
#   - Synthetic data (15B): milestone 3b — agentic traces not yet generated

ALL_SOURCES = list(STACK_V2_LANGS.keys()) + [
    # Code
    "jupyter",
    "rosetta_code",
    # Q&A
    "stackoverflow",
    "stack_exchange_other",
    # Commits / issues
    "github_commits",
    "github_issues",
    # Reference — HF
    "wikibooks",
    # Reference — direct download
    "tldr_pages",
    "man_pages",
    "python_docs",
    "peps",
    "rfcs",
    # NL / general knowledge
    "fineweb_edu",
    "arxiv",
    "wikipedia",
    # Instruction following
    "flan_v2",
    "natural_instructions",
    "openhermes",
    "nl2bash",
    # Math
    "numinamath",
    "competition_math",
    "proof_pile",
    # Pedagogical + papers
    "papers_with_code",
    "pypi_readmes",
    "fastai_notebooks",
    "python_ds_handbook",
    "sicp",
]

# Sources dispatched via direct-download functions (no DataTrove executor)
_DIRECT_SOURCES = {
    "nl2bash", "tldr_pages", "man_pages", "python_docs", "peps", "rfcs",
    "papers_with_code", "pypi_readmes", "fastai_notebooks", "python_ds_handbook", "sicp",
}

def main():
    parser = argparse.ArgumentParser(description="Corpus pipeline: download + format")
    parser.add_argument("--out",     required=True)
    parser.add_argument("--logs",    default="logs/pipeline")
    parser.add_argument("--source",  choices=ALL_SOURCES, action="append", dest="sources")
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--full",    action="store_true")
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_dir  = Path(args.out)
    logs_dir = Path(args.logs)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    sources = ALL_SOURCES if args.all else (args.sources or [])
    if not sources:
        parser.print_help()
        sys.exit(1)

    for source in sources:
        print(f"\n{'='*60}\n{source}  (full={args.full}, workers={args.workers})\n{'='*60}")

        # Direct-download sources
        if source == "nl2bash":
            _nl2bash_pipeline(out_dir, logs_dir)
            continue
        if source == "tldr_pages":
            _tldr_pages_pipeline(out_dir, logs_dir)
            continue
        if source == "man_pages":
            _man_pages_pipeline(out_dir, logs_dir)
            continue
        if source == "python_docs":
            _python_docs_pipeline(out_dir, logs_dir)
            continue
        if source == "peps":
            _peps_pipeline(out_dir, logs_dir)
            continue
        if source == "rfcs":
            _rfcs_pipeline(out_dir, logs_dir)
            continue
        if source == "papers_with_code":
            _papers_with_code_pipeline(out_dir, logs_dir, limit=args.limit or EXPERIMENT_CAPS.get("papers_with_code"))
            continue
        if source == "pypi_readmes":
            _pypi_readmes_pipeline(out_dir, logs_dir, limit=args.limit)
            continue
        if source == "fastai_notebooks":
            _fastai_notebooks_pipeline(out_dir, logs_dir)
            continue
        if source == "python_ds_handbook":
            _python_ds_handbook_pipeline(out_dir, logs_dir)
            continue
        if source == "sicp":
            _sicp_pipeline(out_dir, logs_dir)
            continue

        # DataTrove executor sources
        kw = dict(out_dir=out_dir, full=args.full, workers=args.workers,
                  logs=logs_dir, limit_override=args.limit)
        if source in STACK_V2_LANGS:
            executor = _stack_v2_pipeline(source, **kw)
        elif source == "jupyter":
            executor = _jupyter_pipeline(**kw)
        elif source == "rosetta_code":
            executor = _rosetta_code_pipeline(**kw)
        elif source == "stackoverflow":
            executor = _stackoverflow_pipeline(**kw)
        elif source == "stack_exchange_other":
            executor = _stack_exchange_other_pipeline(**kw)
        elif source == "github_commits":
            executor = _github_commits_pipeline(**kw)
        elif source == "github_issues":
            executor = _github_issues_pipeline(**kw)
        elif source == "wikibooks":
            executor = _wikibooks_pipeline(**kw)
        elif source == "fineweb_edu":
            executor = _fineweb_edu_pipeline(**kw)
        elif source == "arxiv":
            executor = _arxiv_pipeline(**kw)
        elif source == "wikipedia":
            executor = _wikipedia_pipeline(**kw)
        elif source == "flan_v2":
            executor = _flan_v2_pipeline(**kw)
        elif source == "natural_instructions":
            executor = _natural_instructions_pipeline(**kw)
        elif source == "openhermes":
            executor = _openhermes_pipeline(**kw)
        elif source == "numinamath":
            executor = _numinamath_pipeline(**kw)
        elif source == "competition_math":
            executor = _competition_math_pipeline(**kw)
        elif source == "proof_pile":
            executor = _proof_pile_pipeline(**kw)
        else:
            print(f"  {source}: not implemented")
            continue
        executor.run()


if __name__ == "__main__":
    main()
