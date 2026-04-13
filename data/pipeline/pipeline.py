"""
Corpus pipeline: download + format per source, no filtering/dedup.
Sources match docs/dataset.md v1 target (experiment-scale caps applied).

Usage:
    python data/pipeline/pipeline.py --source stack_v2_python --out data/processed/
    python data/pipeline/pipeline.py --all --out data/processed/
    python data/pipeline/pipeline.py --all --full --workers 16 --out data/processed/
"""

import argparse
import bz2
import gzip
import io
import json
import re
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

# Some Modal container images lack up-to-date CA bundles — use an unverified
# context only for hosts where we've confirmed this is the fallback needed.
_SSL_UNVERIFIED = ssl.create_default_context()
_SSL_UNVERIFIED.check_hostname = False
_SSL_UNVERIFIED.verify_mode = ssl.CERT_NONE

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

MAX_FILE_BYTES = 1_000_000

# DataTrove compat: adapters must NOT return None — some versions crash on
# parsed_data.get() when adapter returns None.  Return {"text": ""} to skip;
# DataTrove filters out empty-text documents internally.
_SKIP = {"text": ""}

# ── experiment caps (document count) ─────────────────────────────────────────
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
    "rosetta_code":            None,   # tiny ~1K — always full
    # Q&A
    "stackoverflow":        200_000,
    "stack_exchange_other": 100_000,
    # Commits / issues
    "github_commits":       150_000,
    "github_issues":        100_000,
    # Reference — direct download (all small, always full unless overridden)
    "wikibooks":             30_000,
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
    "nl2bash":                 None,
    # Math
    "numinamath":            50_000,
    "competition_math":      30_000,
    "proof_pile":            50_000,
    # Pedagogical
    "papers_with_code":      50_000,
    "pypi_readmes":          50_000,
    "fastai_notebooks":        None,
    "python_ds_handbook":      None,
    "sicp":                    None,
    # Math — DeepMind
    "deepmind_math":         100_000,
    # Reference — tech docs (Git book + Docker docs + Bash manual)
    "tech_docs":               None,
    # Reference — library docs (NumPy, Pandas, sklearn, matplotlib, requests)
    "library_docs":            None,
}

# ── Stack v2 ──────────────────────────────────────────────────────────────────

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

def _stack_v2_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    content = data.get("content") or ""
    if not content:
        return _SKIP
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

def _jupyter_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    # codeparrot/github-jupyter-parsed schema: cells (list[str]), types (list[str])
    cells = data.get("cells") or []
    types = data.get("types") or []
    if not cells:
        return _SKIP
    parts = []
    for cell, ctype in zip(cells, types) if types else zip(cells, [""] * len(cells)):
        cell = cell.strip() if isinstance(cell, str) else ""
        if cell:
            parts.append(cell)
    text = "\n\n".join(parts).strip()
    if not text:
        return _SKIP
    return {
        "text": text[:MAX_FILE_BYTES],
        "id":   str(id_in_file),
        "metadata": {},
    }

def _jupyter_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["jupyter"])
    limit = cap if cap is not None else -1
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

def _rosetta_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    task = (data.get("task") or "").strip()
    lang = (data.get("language") or "").strip()
    code = (data.get("code") or "").strip()
    if not code:
        return _SKIP
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


# ── Stack Overflow ────────────────────────────────────────────────────────────

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

def _so_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    answers = data.get("answers") or []
    if not answers:
        return _SKIP
    best = max(answers, key=lambda a: (a.get("selected", False), a.get("pm_score", 0)))
    answer = _strip_html(best.get("text", "")).strip()
    if not answer:
        return _SKIP
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


# ── Stack Exchange (other sites) ──────────────────────────────────────────────

def _se_other_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    instruction = (data.get("instruction") or data.get("question") or "").strip()
    response    = (data.get("response") or data.get("answer") or "").strip()
    if not response:
        return _SKIP
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
            dataset_options={"split": "test"},
            adapter=_se_other_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "stack_exchange_other")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "stack_exchange_other"))


# ── GitHub commits ────────────────────────────────────────────────────────────

def _commits_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    subject = (data.get("subject") or "").strip()
    content = (data.get("new_contents") or "").strip()
    if not content:
        return _SKIP
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


# ── GitHub issues ─────────────────────────────────────────────────────────────

_ISSUE_TOKENS = re.compile(r"<issue_start>|<issue_comment>|<issue_closed>|<issue_opened>")

def _issues_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    if data.get("pull_request"):
        return _SKIP
    content = (data.get("content") or "").strip()
    if not content:
        return _SKIP
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
# Direct download from Wikimedia XML dumps (wikimedia/wikibooks does not exist on HF).

def _wikibooks_pipeline(out_dir: Path, logs: Path, limit=None):
    import xml.etree.ElementTree as ET

    out_path  = out_dir / "wikibooks"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    empty_flag = out_path / "_empty"  # written if previous run produced 0 docs (bug)
    if done_flag.exists() and not empty_flag.exists():
        print("wikibooks: already done, skipping")
        return
    # Clear stale flags so we retry
    for f in (done_flag, empty_flag):
        if f.exists():
            f.unlink()

    url = "https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2"
    try:
        print(f"wikibooks: downloading {url} ...")
        with urllib.request.urlopen(url, timeout=300) as resp:
            compressed = resp.read()
    except Exception as e:
        print(f"wikibooks: download failed: {e} — skipping")
        done_flag.touch()
        return

    cap = limit if limit is not None else EXPERIMENT_CAPS.get("wikibooks")
    out_file = out_path / "wikibooks.jsonl.gz"
    n = 0

    def _local(tag):
        """Strip XML namespace prefix: {ns}tag → tag"""
        return tag.split("}", 1)[-1] if "}" in tag else tag

    with gzip.open(out_file, "wt") as f:
        xml_bytes = bz2.decompress(compressed)
        for event, elem in ET.iterparse(io.BytesIO(xml_bytes), events=("end",)):
            if _local(elem.tag) != "page":
                continue
            if cap and n >= cap:
                break
            ns_text    = next((c.text for c in elem.iter() if _local(c.tag) == "ns"), None)
            if ns_text != "0":
                elem.clear()
                continue
            title_text = next((c.text for c in elem.iter() if _local(c.tag) == "title"), None)
            wikitext   = next((c.text for c in elem.iter() if _local(c.tag) == "text"), None)
            title = (title_text or "").strip()
            text  = (wikitext  or "").strip()
            if text:
                body = f"# {title}\n\n{text}" if title else text
                f.write(json.dumps({"text": body[:MAX_FILE_BYTES], "id": title or str(n)}) + "\n")
                n += 1
            elem.clear()

    if n == 0:
        empty_flag.touch()
        print(f"wikibooks: WARNING — wrote 0 articles; will retry next run")
    else:
        done_flag.touch()
        print(f"wikibooks: wrote {n} articles to {out_file}")


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
                parts = name.split("/")
                if len(parts) < 2:
                    continue
                section = parts[-2]
                if not re.match(r"^man[1237]$", section):
                    continue
                text = zf.read(name).decode("utf-8", errors="replace").strip()
                if text:
                    f.write(json.dumps({"text": text, "id": name}) + "\n")
                    n += 1
    done_flag.touch()
    print(f"man_pages: wrote {n} pages to {out_file}")


# ── Python docs ───────────────────────────────────────────────────────────────

def _python_docs_pipeline(out_dir: Path, logs: Path):
    import tarfile
    out_path  = out_dir / "python_docs"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("python_docs: already done, skipping")
        return

    # Try recent Python versions in order; URL uses version-specific path
    content = None
    for ver in ("3.13", "3.12", "3.11", "3.10"):
        url = f"https://docs.python.org/{ver}/archives/python-{ver}-docs-text.tar.bz2"
        try:
            print(f"python_docs: trying {url} ...")
            with urllib.request.urlopen(url, timeout=120) as resp:
                content = resp.read()
            print(f"python_docs: got {len(content)//1024}KB from {url}")
            break
        except Exception as e:
            print(f"python_docs: {url} failed: {e}")

    if not content:
        print("python_docs: all URLs failed — skipping")
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
    _pep_re  = re.compile(r"pep-\d{4}\.rst$")
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

_KEY_RFCS = [
    7230, 7231, 7232, 7233, 7234, 7235,  # HTTP/1.1
    7540,                                 # HTTP/2
    9110, 9111, 9112,                     # HTTP semantics (latest)
    8446,                                 # TLS 1.3
    8259, 7159, 6901, 6902,               # JSON
    1034, 1035,                           # DNS
    5321,                                 # SMTP
    793, 791,                             # TCP/IP
    6455,                                 # WebSocket
    6749, 7519, 7517,                     # OAuth/JWT
    2045, 2046, 2047, 2048, 2049,         # MIME
    3986,                                 # URI
    4251, 4252, 4253,                     # SSH
    2616,                                 # HTTP/1.1 (original, widely cited)
    2119,                                 # MUST/SHOULD/MAY keywords
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

def _fineweb_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    text = (data.get("text") or "").strip()
    if not text:
        return _SKIP
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

def _arxiv_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    # ccdv/arxiv-summarization schema: article (body), abstract
    article  = (data.get("article") or "").strip()
    abstract = (data.get("abstract") or "").strip()
    text = f"{abstract}\n\n{article}".strip() if abstract else article
    if not text:
        return _SKIP
    return {
        "text": text[:MAX_FILE_BYTES],
        "id":   str(id_in_file),
        "metadata": {},
    }

def _arxiv_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["arxiv"])
    limit = cap if cap is not None else -1
    # ccdv/arxiv-summarization: clean Parquet, full-text arxiv papers
    # scientific_papers, peS2o, and RedPajama-Data-1T-Sample all use deprecated scripts
    # or don't exist on HF.
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="ccdv/arxiv-summarization",
            dataset_options={"name": "document", "split": "train"},
            adapter=_arxiv_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "arxiv")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "arxiv"))


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def _wikipedia_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    text = (data.get("text") or "").strip()
    if not text:
        return _SKIP
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

def _flan_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    inp  = (data.get("inputs") or "").strip()
    out  = (data.get("targets") or "").strip()
    if not out:
        return _SKIP
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

def _natural_instructions_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    # Muennighoff/natural-instructions schema: task_name, id, definition, inputs, targets
    inp = (data.get("inputs") or data.get("input") or "").strip()
    targets = data.get("targets") or data.get("output") or []
    if isinstance(targets, str):
        targets = [targets]
    out = targets[0].strip() if targets else ""
    if not out:
        return _SKIP
    definition = (data.get("definition") or data.get("Definition") or "")
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

def _openhermes_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    convs = data.get("conversations") or []
    if not convs:
        return _SKIP
    parts = []
    for turn in convs:
        role  = (turn.get("from") or "").strip()
        value = (turn.get("value") or "").strip()
        if not value:
            continue
        label = "Human" if role == "human" else "Assistant"
        parts.append(f"{label}: {value}")
    if not parts:
        return _SKIP
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

def _numinamath_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    problem  = (data.get("problem") or "").strip()
    solution = (data.get("solution") or "").strip()
    if not problem or not solution:
        return _SKIP
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

def _competition_math_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    # meta-math/MetaMathQA schema: query, response, type
    query    = (data.get("query") or data.get("problem") or "").strip()
    response = (data.get("response") or data.get("solution") or "").strip()
    if not query or not response:
        return _SKIP
    math_type = data.get("type") or data.get("level") or ""
    header    = f"[{math_type}]\n" if math_type else ""
    return {
        "text": f"{header}Problem: {query}\n\nSolution: {response}",
        "id":   str(id_in_file),
        "metadata": {},
    }

def _competition_math_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["competition_math"])
    limit = cap if cap is not None else -1
    # meta-math/MetaMathQA: clean Parquet, 395K augmented math problems
    # lighteval/MATH doesn't exist on Hub
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="meta-math/MetaMathQA",
            dataset_options={"split": "train"},
            adapter=_competition_math_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "competition_math")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "competition_math"))


# ── Proof-Pile 2 ──────────────────────────────────────────────────────────────

def _proof_pile_adapter(self, data: dict, path: str, id_in_file: int) -> dict:
    text = (data.get("text") or "").strip()
    if not text:
        return _SKIP
    return {
        "text": text[:MAX_FILE_BYTES],
        "id":   str(id_in_file),
        "metadata": {},
    }

def _proof_pile_pipeline(out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS["proof_pile"])
    limit = cap if cap is not None else -1
    # open-web-math/open-web-math: ~15B tokens of math web content, clean Parquet
    # EleutherAI/proof-pile-2 uses a deprecated loading script
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="open-web-math/open-web-math",
            dataset_options={"split": "train"},
            adapter=_proof_pile_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "proof_pile")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "proof_pile"))


# ── Papers with Code ──────────────────────────────────────────────────────────

def _papers_with_code_pipeline(out_dir: Path, logs: Path, limit=None):
    out_path  = out_dir / "papers_with_code"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("papers_with_code: already done, skipping")
        return

    # Primary source: J0nasW/paperswithcode on HuggingFace (44K papers, no SSL issues)
    # Fallback: production-media.paperswithcode.com (often blocked by SSL)
    papers = []
    try:
        from datasets import load_dataset
        ds = load_dataset("J0nasW/paperswithcode", data_files="papers_train.csv", split="train")
        papers = list(ds)
        print(f"papers_with_code: loaded {len(papers)} papers from HF (J0nasW/paperswithcode)")
    except Exception as e:
        print(f"papers_with_code: HF load failed: {e} — trying direct download ...")

    if not papers:
        url = "https://production-media.paperswithcode.com/about/papers-with-abstracts.json.gz"
        raw = None
        for ctx in (None, _SSL_UNVERIFIED):
            try:
                print(f"papers_with_code: downloading {url} ...")
                with urllib.request.urlopen(url, context=ctx, timeout=120) as resp:
                    raw = resp.read()
                break
            except Exception as e:
                print(f"papers_with_code: attempt failed: {e}")
        if raw is None:
            print("papers_with_code: all download attempts failed — skipping")
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

def _pypi_readmes_pipeline(out_dir: Path, logs: Path, limit=None):
    out_path  = out_dir / "pypi_readmes"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("pypi_readmes: already done, skipping")
        return

    cap = limit if limit is not None else EXPERIMENT_CAPS.get("pypi_readmes") or 50_000

    top_url = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"
    try:
        print(f"pypi_readmes: fetching package list ...")
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
                info        = data.get("info") or {}
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
                pass
    done_flag.touch()
    print(f"pypi_readmes: wrote {n} READMEs to {out_file}")


# ── GitHub-hosted pedagogical sources ─────────────────────────────────────────

def _github_zip_notebooks(out_dir: Path, source_name: str, zip_url: str,
                           extensions=(".ipynb",), timeout=120):
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
                if name.endswith(".ipynb"):
                    try:
                        nb    = json.loads(raw.decode("utf-8", errors="replace"))
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
    _github_zip_notebooks(
        out_dir, "sicp",
        "https://github.com/sarabander/sicp/archive/refs/heads/master.zip",
        extensions=(".html", ".xml", ".xhtml"),
        timeout=300,
    )


# ── NL2Bash ───────────────────────────────────────────────────────────────────

def _nl2bash_pipeline(out_dir: Path, logs: Path):
    out_path  = out_dir / "nl2bash"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("nl2bash: already done, skipping")
        return

    records = []

    # Attempt 1: Dropbox zip (original source used by jiacheng-ye/nl2bash HF script)
    # Contains nl2bash/train.json + dev.json + test.json with {"nl":..., "bash":...} entries
    try:
        dropbox_url = "https://www.dropbox.com/s/wy7uahzbir7lrq1/nl2bash.zip?dl=1"
        print(f"nl2bash: downloading {dropbox_url} ...")
        with urllib.request.urlopen(dropbox_url, timeout=120) as resp:
            raw = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for split in ("train", "dev", "test"):
                fname = f"nl2bash/{split}.json"
                if fname in zf.namelist():
                    data = json.loads(zf.read(fname))
                    records.extend(data)
        print(f"nl2bash: loaded {len(records)} records from Dropbox zip")
    except Exception as e:
        print(f"nl2bash: Dropbox zip failed: {e}")

    # Attempt 2: HF trust_remote_code (works on older datasets versions)
    if not records:
        try:
            from datasets import load_dataset
            ds = load_dataset("jiacheng-ye/nl2bash", split="train", trust_remote_code=True)
            records = list(ds)
            print(f"nl2bash: loaded {len(records)} records from HF (trust_remote_code)")
        except Exception as e:
            print(f"nl2bash: HF trust_remote_code failed: {e}")

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


# ── DeepMind Mathematics ──────────────────────────────────────────────────────

def _deepmind_math_pipeline(out_dir: Path, logs: Path, limit=None):
    out_path  = out_dir / "deepmind_math"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("deepmind_math: already done, skipping")
        return

    try:
        from datasets import get_dataset_config_names, load_dataset
        configs = get_dataset_config_names("deepmind/math_dataset", trust_remote_code=True)
        print(f"deepmind_math: {len(configs)} configs")
    except Exception as e:
        print(f"deepmind_math: failed to get configs: {e} — skipping")
        done_flag.touch()
        return

    cap = limit if limit is not None else EXPERIMENT_CAPS.get("deepmind_math")
    out_file = out_path / "deepmind_math.jsonl.gz"
    n = 0
    with gzip.open(out_file, "wt") as f:
        for config in configs:
            if cap and n >= cap:
                break
            try:
                ds = load_dataset("deepmind/math_dataset", config,
                                  split="train", trust_remote_code=True, streaming=True)
                for ex in ds:
                    if cap and n >= cap:
                        break
                    q = (ex.get("question") or "").strip()
                    a = (ex.get("answer") or "").strip()
                    if q and a:
                        text = f"Problem: {q}\n\nSolution: {a}"
                        f.write(json.dumps({"text": text[:MAX_FILE_BYTES],
                                            "id":   f"{config}_{n}"}) + "\n")
                        n += 1
            except Exception as e:
                print(f"deepmind_math: config {config} failed: {e}")
    done_flag.touch()
    print(f"deepmind_math: wrote {n} problems to {out_file}")


# ── Tech docs (Pro Git + Docker docs + Bash manual) ───────────────────────────

def _tech_docs_pipeline(out_dir: Path, logs: Path):
    out_path  = out_dir / "tech_docs"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("tech_docs: already done, skipping")
        return

    out_file = out_path / "tech_docs.jsonl.gz"
    n = 0
    with gzip.open(out_file, "wt") as f:

        # 1. Pro Git book (AsciiDoc)
        n_git = 0
        try:
            url = "https://github.com/progit/progit2/archive/refs/heads/master.zip"
            print(f"tech_docs: downloading Pro Git book ...")
            with urllib.request.urlopen(url, timeout=120) as resp:
                content = resp.read()
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for name in sorted(zf.namelist()):
                    if not name.endswith(".adoc"):
                        continue
                    if any(x in name for x in ["/images/", "/_"]):
                        continue
                    text = zf.read(name).decode("utf-8", errors="replace").strip()
                    if len(text) < 200:
                        continue
                    f.write(json.dumps({"text": text[:MAX_FILE_BYTES],
                                        "id":   f"progit/{name}"}) + "\n")
                    n_git += 1
            print(f"tech_docs: Pro Git: {n_git} docs")
        except Exception as e:
            print(f"tech_docs: Pro Git failed: {e}")
        n += n_git

        # 2. GNU Bash manual (plain text)
        n_bash = 0
        try:
            url = "https://www.gnu.org/software/bash/manual/bash.txt"
            print(f"tech_docs: downloading Bash manual ...")
            text = None
            for ctx in (None, _SSL_UNVERIFIED):
                try:
                    with urllib.request.urlopen(url, context=ctx, timeout=60) as resp:
                        text = resp.read().decode("utf-8", errors="replace").strip()
                    break
                except Exception:
                    pass
            if text:
                sections = [s.strip() for s in text.split("\x0c") if len(s.strip()) > 200]
                if not sections:
                    sections = [text]
                for i, sec in enumerate(sections):
                    f.write(json.dumps({"text": sec[:MAX_FILE_BYTES],
                                        "id":   f"bash_manual_{i}"}) + "\n")
                    n_bash += 1
            print(f"tech_docs: Bash manual: {n_bash} sections")
        except Exception as e:
            print(f"tech_docs: Bash manual failed: {e}")
        n += n_bash

        # 3. Docker docs (Markdown) — cap download at 150 MB to avoid huge binary assets
        n_docker = 0
        try:
            url = "https://github.com/docker/docs/archive/refs/heads/main.zip"
            print(f"tech_docs: downloading Docker docs (up to 150 MB) ...")
            chunks, total = [], 0
            with urllib.request.urlopen(url, timeout=300) as resp:
                while total < 150 * 1024 * 1024:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
            content = b"".join(chunks)
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for name in sorted(zf.namelist()):
                    if not name.endswith(".md"):
                        continue
                    if any(x in name for x in ["/node_modules/", "/vendor/", "/_"]):
                        continue
                    raw = zf.read(name)
                    if len(raw) < 200:
                        continue
                    text = raw.decode("utf-8", errors="replace").strip()
                    f.write(json.dumps({"text": text[:MAX_FILE_BYTES],
                                        "id":   f"docker/{name}"}) + "\n")
                    n_docker += 1
            print(f"tech_docs: Docker docs: {n_docker} files")
        except Exception as e:
            print(f"tech_docs: Docker docs failed: {e}")
        n += n_docker

    done_flag.touch()
    print(f"tech_docs: wrote {n} docs total to {out_file}")


# ── Library docs (NumPy, Pandas, scikit-learn, Matplotlib, Requests) ──────────

_LIBRARY_REPOS = [
    # (name, org/repo, doc_subdir_prefix)
    ("numpy",       "numpy/numpy",                  "doc/source"),
    ("pandas",      "pandas-dev/pandas",             "doc/source"),
    ("sklearn",     "scikit-learn/scikit-learn",     "doc"),
    ("matplotlib",  "matplotlib/matplotlib",         "doc"),
    ("requests",    "psf/requests",                  "docs"),
]

def _library_docs_pipeline(out_dir: Path, logs: Path):
    out_path  = out_dir / "library_docs"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("library_docs: already done, skipping")
        return

    out_file = out_path / "library_docs.jsonl.gz"
    n = 0
    with gzip.open(out_file, "wt") as f:
        for lib_name, repo, doc_prefix in _LIBRARY_REPOS:
            lib_n = 0
            for branch in ("main", "master"):
                url = f"https://github.com/{repo}/archive/refs/heads/{branch}.zip"
                try:
                    print(f"library_docs: downloading {lib_name} ({branch}) ...")
                    chunks, total = [], 0
                    with urllib.request.urlopen(url, timeout=180) as resp:
                        while total < 80 * 1024 * 1024:  # cap at 80 MB per lib
                            chunk = resp.read(512 * 1024)
                            if not chunk:
                                break
                            chunks.append(chunk)
                            total += len(chunk)
                    content = b"".join(chunks)
                    with zipfile.ZipFile(io.BytesIO(content)) as zf:
                        for fname in sorted(zf.namelist()):
                            if not (fname.endswith(".rst") or fname.endswith(".md")):
                                continue
                            # Confirm it lives under the docs subdirectory
                            parts = fname.split("/", 2)
                            if len(parts) < 3 or not parts[2].startswith(doc_prefix):
                                continue
                            raw = zf.read(fname)
                            if len(raw) < 100:
                                continue
                            text = raw.decode("utf-8", errors="replace").strip()
                            f.write(json.dumps({"text": text[:MAX_FILE_BYTES],
                                                "id":   f"{lib_name}/{fname}"}) + "\n")
                            lib_n += 1
                    break  # success — don't try other branch
                except Exception as e:
                    print(f"library_docs: {lib_name} ({branch}) failed: {e}")
            print(f"library_docs: {lib_name}: {lib_n} docs")
            n += lib_n

    done_flag.touch()
    print(f"library_docs: wrote {n} docs total to {out_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

# Not yet implemented (no clean public dataset):
#   - Dev.to + HashNode (2B): no HF dataset; FineWeb-Edu likely covers most of it
#   - Math Jupyter notebooks (0.5B): covered by `jupyter` source
#   - Synthetic data (15B): milestone 3b — not yet generated

ALL_SOURCES = list(STACK_V2_LANGS.keys()) + [
    "jupyter", "rosetta_code",
    "stackoverflow", "stack_exchange_other",
    "github_commits", "github_issues",
    "wikibooks", "tldr_pages", "man_pages", "python_docs", "peps", "rfcs",
    "fineweb_edu", "arxiv", "wikipedia",
    "flan_v2", "natural_instructions", "openhermes", "nl2bash",
    "numinamath", "competition_math", "proof_pile",
    "papers_with_code", "pypi_readmes", "fastai_notebooks", "python_ds_handbook", "sicp",
    "deepmind_math", "tech_docs", "library_docs",
]

_DIRECT_SOURCES = {
    "wikibooks", "nl2bash", "tldr_pages", "man_pages", "python_docs", "peps", "rfcs",
    "papers_with_code", "pypi_readmes", "fastai_notebooks", "python_ds_handbook", "sicp",
    "deepmind_math", "tech_docs", "library_docs",
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

        if source == "wikibooks":
            _wikibooks_pipeline(out_dir, logs_dir, limit=args.limit or EXPERIMENT_CAPS.get("wikibooks"))
            continue
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
        if source == "deepmind_math":
            _deepmind_math_pipeline(out_dir, logs_dir, limit=args.limit or EXPERIMENT_CAPS.get("deepmind_math"))
            continue
        if source == "tech_docs":
            _tech_docs_pipeline(out_dir, logs_dir)
            continue
        if source == "library_docs":
            _library_docs_pipeline(out_dir, logs_dir)
            continue

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
