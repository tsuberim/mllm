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
import json
import re
import sys
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import Document

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
    # Reference
    "tldr_pages":              None,   # tiny ~7K docs — always full
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
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="bigcode/starcoderdata",
            dataset_options={"name": "jupyter-scripts-dedup-filtered", "split": "train"},
            adapter=_jupyter_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / "jupyter")),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / "jupyter"))


# ── Rosetta Code ──────────────────────────────────────────────────────────────
# cakiki/rosetta-code: task + language + code triples.

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
# ArmelR/stack-exchange-instruction covers Code Review, Unix, ServerFault,
# AskUbuntu, SoftwareEngineering, DevOps, DataScience, etc.

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


# ── tldr-pages ────────────────────────────────────────────────────────────────
# Direct download from GitHub; no HF dataset available.

def _tldr_pages_pipeline(out_dir: Path, logs: Path):
    out_path  = out_dir / "tldr_pages"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("tldr_pages: already done, skipping")
        return

    import urllib.request, zipfile, io
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


# ── FineWeb-Edu ───────────────────────────────────────────────────────────────
# HuggingFaceFW/fineweb-edu: high-quality educational web text (score ≥ 4).

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
# scientific_papers (arxiv config): abstract + full article body.

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
# wikimedia/wikipedia: English Wikipedia (2023-11-01 snapshot).

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
# Muennighoff/flan: aggregated FLAN v2 instruction-following mix.

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
# allenai/natural_instructions: 1600+ task types, input/output pairs.

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
# teknium/OpenHermes-2.5: Mistral-generated instruction-following conversations.

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
# AI-MO/NuminaMath-CoT: competition math problems with chain-of-thought solutions.

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
# lighteval/MATH: Hendrycks MATH benchmark problems with worked solutions.
# Proxies for "DeepMind Mathematics" slot in dataset.md (synthetic, broad coverage).

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
# EleutherAI/proof-pile-2: 55B-token math pretraining corpus; taking ~5% subset.
# Using the arxiv-math subset for maximum signal density.

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


# ── nl2bash ───────────────────────────────────────────────────────────────────
# jiacheng-ye/nl2bash uses a legacy HF loading script. Fetch raw data directly.

def _nl2bash_pipeline(out_dir: Path, logs: Path):
    out_path  = out_dir / "nl2bash"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("nl2bash: already done, skipping")
        return

    import urllib.request
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
                import io, pyarrow.parquet as pq
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

# Sources NOT yet implemented (not on HuggingFace or require custom scraping):
#   - PyPI package READMEs (no clean HF dataset)
#   - Papers with Code (paperswithcode.com API, not on HF)
#   - Man pages, Python stdlib docs, PEPs, RFC docs (require direct download/scraping)
#   - Dev.to, HashNode, Wikibooks, Fast.ai, SICP (web scraping)
#   - Math Jupyter notebooks (no unified HF dataset)
#   - Synthetic data (agentic traces — milestone 3b, not yet generated)

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
    # Reference (direct download)
    "tldr_pages",
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
]

# Sources handled via direct download (no DataTrove executor)
_DIRECT_SOURCES = {"nl2bash", "tldr_pages"}

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
