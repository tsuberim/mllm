"""
DataTrove-based corpus pipeline: download + format in one pass, per source.

No quality filtering or dedup — just download, format text, write JSONL.
Resumable: completed shards tracked in --logs dir; re-run to continue.

Usage:
    python data/pipeline/pipeline.py --source stack_python --out data/processed/
    python data/pipeline/pipeline.py --all --full --workers 16 --out data/processed/
"""

import argparse
import gzip
import json
import re
import sys
import urllib.request
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.data import Document


# ── experiment caps (document count) ─────────────────────────────────────────

EXPERIMENT_CAPS = {
    "stack_python":   1_000_000,
    "stack_bash":       250_000,
    "stack_md":         200_000,
    "stackoverflow":    600_000,
    "github_commits":   200_000,
    "github_issues":    150_000,
    "jupyter":           50_000,
    "nl2bash":             None,  # ~9.3K pairs — always full
}

MAX_FILE_BYTES = 1_000_000

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


# ── stack sources ─────────────────────────────────────────────────────────────

def _stack_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    content = data.get("content") or ""
    if not content:
        return None
    return {
        "text": content[:MAX_FILE_BYTES],
        "id":   data.get("hexsha") or f"{path}/{id_in_file}",
        "metadata": {},
    }

def _stack_pipeline(lang: str, source: str, out_dir: Path, full: bool, workers: int, logs: Path, limit_override=None):
    data_dir = {"python": "data/python", "shell": "data/shell", "markdown": "data/markdown"}[lang]
    cap   = limit_override if limit_override is not None else (None if full else EXPERIMENT_CAPS[source])
    limit = cap if cap is not None else -1
    pipeline = [
        HuggingFaceDatasetReader(
            dataset="bigcode/the-stack-dedup",
            dataset_options={"data_dir": data_dir, "split": "train"},
            adapter=_stack_adapter,
            streaming=True,
            limit=limit,
        ),
        JsonlWriter(output_folder=str(out_dir / source)),
    ]
    return LocalPipelineExecutor(pipeline=pipeline, tasks=workers, logging_dir=str(logs / source))


# ── stack overflow ────────────────────────────────────────────────────────────

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


# ── jupyter notebooks ─────────────────────────────────────────────────────────

def _jupyter_adapter(self, data: dict, path: str, id_in_file: int) -> dict | None:
    try:
        # codeparrot/github-jupyter-parsed stores the notebook as raw JSON in "content"
        raw = data.get("cells") or data.get("content") or ""
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                cells = parsed.get("cells") or [] if isinstance(parsed, dict) else []
            except Exception:
                return None
        else:
            cells = raw if isinstance(raw, list) else []
        parts = []
        for cell in cells:
            try:
                if isinstance(cell, str):
                    cell = json.loads(cell)
                if not isinstance(cell, dict):
                    continue
                ctype = cell.get("cell_type") or ""
                src_raw = cell.get("source") or ""
                src = ("".join(src_raw) if isinstance(src_raw, list) else str(src_raw)).strip()
                if not src:
                    continue
                if ctype == "code":
                    parts.append(src)
                    for out in (cell.get("outputs") or []):
                        try:
                            if not isinstance(out, dict):
                                continue
                            text = out.get("text") or (out.get("data") or {}).get("text/plain") or []
                            if isinstance(text, list):
                                text = "".join(text)
                            text = str(text).strip()
                            if text:
                                parts.append("# output:\n# " + "\n# ".join(text.splitlines()))
                        except Exception:
                            continue
                else:
                    parts.append(src)
            except Exception:
                continue
        if not parts:
            return None
        return {
            "text": "\n\n".join(parts)[:MAX_FILE_BYTES],
            "id":   data.get("id") or f"{path}/{id_in_file}",
            "metadata": {},
        }
    except Exception:
        return None

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


# ── nl2bash ───────────────────────────────────────────────────────────────────
# jiacheng-ye/nl2bash uses a legacy HF loading script that newer datasets
# versions reject. Fetch the raw JSONL directly from the HF datasets API.

def _nl2bash_pipeline(out_dir: Path, logs: Path):
    """Download NL2Bash and write to JSONL, bypassing DataTrove.

    jiacheng-ye/nl2bash uses a legacy HF loading script that newer datasets
    versions reject. We fetch the raw TSV/JSON from the original GitHub repo.
    Non-fatal: if all attempts fail, skip with a warning (it's only ~9K pairs).
    """
    out_path = out_dir / "nl2bash"
    out_path.mkdir(parents=True, exist_ok=True)
    done_flag = out_path / "_done"
    if done_flag.exists():
        print("nl2bash: already done, skipping")
        return

    # Raw data from the original TellinaTool/nl2bash GitHub repo
    urls = [
        # invocations file: one JSON object per line with "invocation" and "cmd_str"
        "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash_scripts.json",
        "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/all_invocations.json",
        # HF parquet shard
        "https://huggingface.co/datasets/jiacheng-ye/nl2bash/resolve/main/data/train-00000-of-00001.parquet",
    ]

    records = []
    for url in urls:
        try:
            print(f"nl2bash: trying {url} ...")
            with urllib.request.urlopen(url, timeout=30) as resp:
                content = resp.read()
            if url.endswith(".parquet"):
                import io
                import pyarrow.parquet as pq
                table = pq.read_table(io.BytesIO(content))
                df = table.to_pydict()
                nls   = df.get("nl") or df.get("invocation") or []
                bashes = df.get("bash") or df.get("cmd") or []
                records = [{"nl": n, "bash": b} for n, b in zip(nls, bashes)]
            else:
                raw = json.loads(content)
                if isinstance(raw, list):
                    records = raw
                elif isinstance(raw, dict):
                    # {id: {invocation: ..., cmd_str: ...}, ...}
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
        print("nl2bash: WARNING — all fetch methods failed, skipping (dataset is optional)")
        done_flag.touch()
        return

    out_file = out_path / "nl2bash.jsonl.gz"
    n = 0
    with gzip.open(out_file, "wt") as f:
        for r in records:
            nl  = (r.get("nl") or r.get("cmd_str") or r.get("invocation_desc") or "").strip()
            cmd = (r.get("bash") or r.get("invocation") or r.get("cmd") or "").strip()
            if nl and cmd:
                f.write(json.dumps({"text": f"# {nl}\n{cmd}", "id": nl[:40]}) + "\n")
                n += 1

    done_flag.touch()
    print(f"nl2bash: wrote {n} pairs to {out_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

ALL_SOURCES = [
    "stack_python", "stack_bash", "stack_md",
    "stackoverflow", "github_commits", "github_issues",
    "jupyter", "nl2bash",
]

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
        if source == "stack_python":
            executor = _stack_pipeline("python",   source, out_dir, args.full, args.workers, logs_dir, args.limit)
        elif source == "stack_bash":
            executor = _stack_pipeline("shell",    source, out_dir, args.full, args.workers, logs_dir, args.limit)
        elif source == "stack_md":
            executor = _stack_pipeline("markdown", source, out_dir, args.full, args.workers, logs_dir, args.limit)
        elif source == "stackoverflow":
            executor = _stackoverflow_pipeline(out_dir, args.full, args.workers, logs_dir, args.limit)
        elif source == "github_commits":
            executor = _github_commits_pipeline(out_dir, args.full, args.workers, logs_dir, args.limit)
        elif source == "github_issues":
            executor = _github_issues_pipeline(out_dir, args.full, args.workers, logs_dir, args.limit)
        elif source == "jupyter":
            executor = _jupyter_pipeline(out_dir, args.full, args.workers, logs_dir, args.limit)
        elif source == "nl2bash":
            _nl2bash_pipeline(out_dir, logs_dir)
            continue
        else:
            print(f"  {source}: not implemented")
            continue
        executor.run()


if __name__ == "__main__":
    main()
