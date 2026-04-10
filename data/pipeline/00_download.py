"""
Download and filter raw corpus sources to JSONL shards.

Sources:
  - The Stack (dedup): Python, Bash/Shell, Markdown (Apache 2.0 / permissive licenses)
  - Stack Exchange Q&A (CC BY-SA 4.0) — HuggingFaceH4/stack-exchange-preferences
  - GitHub commits (bigcode/commitpackft) — Python + Shell, Apache 2.0
  - GitHub issues (bigcode/the-stack-github-issues) — from permissive-licensed repos
  - tldr-pages (MIT)

Experiment-scale defaults (--max-records) target ~1.5B total tokens proportional
to the full corpus distribution. Omit --max-records for full-scale runs.

Usage:
    python data/pipeline/00_download.py --source stack_python  --out data/raw/
    python data/pipeline/00_download.py --source stack_bash    --out data/raw/
    python data/pipeline/00_download.py --source stack_md      --out data/raw/
    python data/pipeline/00_download.py --source stackoverflow --out data/raw/
    python data/pipeline/00_download.py --source github_commits --out data/raw/
    python data/pipeline/00_download.py --source github_issues  --out data/raw/
    python data/pipeline/00_download.py --source tldr           --out data/raw/
    python data/pipeline/00_download.py --all --out data/raw/
    # Stack Overflow via legacy XML dump (optional; HF source preferred):
    python data/pipeline/00_download.py --source stackoverflow --so-xml /path/to/Posts.xml --out data/raw/
"""

import argparse
import gzip
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# HTML helpers (for Stack Exchange content)
# ---------------------------------------------------------------------------

_HTML_TAG     = re.compile(r"<[^>]+>")
_HTML_ENTITY  = re.compile(r"&(amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);")
_ENTITY_MAP   = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "nbsp": " "}

def _strip_html(text: str) -> str:
    text = _HTML_TAG.sub("", text)
    def _ent(m):
        name = m.group(1)
        if name in _ENTITY_MAP:
            return _ENTITY_MAP[name]
        if name.startswith("#x"):
            return chr(int(name[2:], 16))
        if name.startswith("#"):
            return chr(int(name[1:]))
        return " "
    text = _HTML_ENTITY.sub(_ent, text).strip()
    # Drop surrogate characters (malformed HTML) — they can't be encoded as UTF-8
    return text.encode("utf-8", errors="replace").decode("utf-8")

def _has_code(text: str) -> bool:
    """True if text contains a code block (markdown or HTML)."""
    return bool(re.search(r"```|^    \S|\t\S|<code>|<pre>", text, re.MULTILINE))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SHARD_SIZE = 500_000      # records per JSONL shard
MAX_FILE_BYTES = 1_000_000  # 1MB — skip pathological files

# Experiment-scale caps (omit --max-records to use these; pass --max-records=0 for full run)
_STACK_PYTHON_EXPERIMENT_MAX  = 1_000_000
_STACK_BASH_EXPERIMENT_MAX    =   250_000
_STACK_MD_EXPERIMENT_MAX      =   200_000
_SO_EXPERIMENT_MAX            =   600_000
_COMMIT_EXPERIMENT_MAX        =   200_000
_ISSUES_EXPERIMENT_MAX        =   150_000

PERMISSIVE_LICENSES = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
    "ISC", "Unlicense", "CC0-1.0", "WTFPL", "0BSD",
    "BSD-4-Clause", "AFL-3.0", "ECL-2.0",
}

SO_TAGS = {
    "bash", "shell", "python", "python-3.x", "linux", "git",
    "docker", "regex", "awk", "sed", "grep", "command-line",
    "scripting", "terminal", "subprocess", "os.path", "pathlib",
    "argparse", "click", "flask", "fastapi", "django",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_shard(out_dir: Path, source: str, idx: int):
    path = out_dir / f"{source}_{idx:04d}.jsonl"
    return open(path, "w")


class ShardWriter:
    def __init__(self, out_dir: Path, source: str):
        self.out_dir = out_dir
        self.source = source
        self.shard_idx = 0
        self.count = 0
        self.total = 0
        self._f = open_shard(out_dir, source, 0)

    def write(self, record: dict):
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1
        self.total += 1
        if self.count >= SHARD_SIZE:
            self._f.close()
            self.shard_idx += 1
            self._f = open_shard(self.out_dir, self.source, self.shard_idx)
            self.count = 0

    def close(self):
        self._f.close()
        print(f"  {self.source}: {self.total:,} records → {self.shard_idx + 1} shards")


# ---------------------------------------------------------------------------
# The Stack v2
# ---------------------------------------------------------------------------

def _stack_lang_filter(row) -> bool:
    """Keep files with permissive license and reasonable size."""
    licenses = (
        row.get("max_stars_repo_licenses")
        or row.get("max_issues_repo_licenses")
        or row.get("max_forks_repo_licenses")
        or []
    )
    if not any(lic in PERMISSIVE_LICENSES for lic in licenses):
        return False
    content = row.get("content") or ""
    if not content or len(content) > MAX_FILE_BYTES:
        return False
    if len(content) < 64:
        return False
    return True


def download_stack(out_dir: Path, lang: str, source_name: str, max_records: int = None, default_max: int = None):
    """Stream The Stack (dedup) for a given language, filter, write shards."""
    data_dir = {
        "python": "data/python",
        "shell": "data/shell",
        "markdown": "data/markdown",
    }.get(lang, f"data/{lang}")

    cap = max_records if max_records is not None else default_max
    print(f"Streaming the-stack-dedup — {lang} (cap={cap:,}) ...")
    writer = ShardWriter(out_dir, source_name)
    kept = 0
    seen = 0

    ds = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir=data_dir,
        split="train",
        streaming=True,
    )

    for row in tqdm(ds, desc=lang):
        seen += 1
        if cap and kept >= cap:
            break
        if not _stack_lang_filter(row):
            continue
        licenses = (
            row.get("max_stars_repo_licenses")
            or row.get("max_issues_repo_licenses")
            or []
        )
        writer.write({
            "source": source_name,
            "content": row["content"],
            "lang": row.get("lang", lang),
            "licenses": licenses,
            "path": row.get("max_stars_repo_path", ""),
            "size": len(row["content"]),
        })
        kept += 1

    writer.close()
    print(f"  kept {kept:,} / {seen:,} ({100*kept/max(seen,1):.1f}%)")


# ---------------------------------------------------------------------------
# Stack Overflow
# ---------------------------------------------------------------------------

def _so_has_target_tag(tags_str: str) -> bool:
    tags = set(re.findall(r"<([^>]+)>", tags_str or ""))
    return bool(tags & SO_TAGS)


def download_stackoverflow(out_dir: Path, posts_xml_path: str = None):
    """
    Parse Stack Overflow Posts.xml dump.
    Download from: https://archive.org/details/stackexchange
    (stackoverflow.com-Posts.7z, then extract Posts.xml)

    If posts_xml_path not provided, prints instructions.
    """
    if not posts_xml_path or not os.path.exists(posts_xml_path):
        print("""
Stack Overflow data dump requires manual download:
  1. Go to: https://archive.org/details/stackexchange
  2. Download: stackoverflow.com-Posts.7z (~22GB)
  3. Extract Posts.xml (~90GB)
  4. Re-run with: --so-xml /path/to/Posts.xml
""")
        return

    print("Parsing Stack Overflow Posts.xml ...")
    writer = ShardWriter(out_dir, "stackoverflow")

    questions = {}   # id -> {title, tags, score}
    kept = 0

    for event, elem in tqdm(ET.iterparse(posts_xml_path, events=("end",))):
        if elem.tag != "row":
            continue

        post_type = elem.get("PostTypeId")
        score = int(elem.get("Score", 0))
        tags = elem.get("Tags", "")
        body = elem.get("Body", "")

        if post_type == "1":  # Question
            if score >= 0 and _so_has_target_tag(tags):
                questions[elem.get("Id")] = {
                    "title": elem.get("Title", ""),
                    "tags": tags,
                    "score": score,
                    "body": body,
                }

        elif post_type == "2":  # Answer
            parent_id = elem.get("ParentId")
            is_accepted = elem.get("AcceptedAnswerId") is not None or elem.get("IsAccepted") == "1"
            if parent_id in questions and (is_accepted or score >= 5):
                q = questions[parent_id]
                writer.write({
                    "source": "stackoverflow",
                    "content": f"Q: {q['title']}\n\n{q['body']}\n\nA: {body}",
                    "tags": q["tags"],
                    "q_score": q["score"],
                    "a_score": score,
                })
                kept += 1

        elem.clear()

    writer.close()
    print(f"  kept {kept:,} Q&A pairs")


# ---------------------------------------------------------------------------
# tldr-pages
# ---------------------------------------------------------------------------

def download_tldr(out_dir: Path, repo_path: str = None):
    """
    Parse tldr-pages markdown files.
    Clone with: git clone https://github.com/tldr-pages/tldr
    """
    if not repo_path or not os.path.exists(repo_path):
        print("""
tldr-pages requires cloning the repo first:
  git clone https://github.com/tldr-pages/tldr /tmp/tldr
  Then re-run with: --tldr-path /tmp/tldr
""")
        return

    print("Processing tldr-pages ...")
    writer = ShardWriter(out_dir, "tldr")

    pages_dir = Path(repo_path) / "pages"
    for md_file in tqdm(sorted(pages_dir.rglob("*.md"))):
        content = md_file.read_text(errors="replace").strip()
        if len(content) < 50:
            continue
        writer.write({
            "source": "tldr",
            "content": content,
            "path": str(md_file.relative_to(repo_path)),
            "size": len(content),
        })

    writer.close()


# ---------------------------------------------------------------------------
# Stack Exchange Q&A  (HuggingFaceH4/stack-exchange-preferences)
# ---------------------------------------------------------------------------

# Default cap for experiment-scale runs (~145M tokens at ~2k chars/rec)
_SO_EXPERIMENT_MAX = 600_000

def download_stackoverflow_hf(out_dir: Path, max_records: int = None):
    """
    Stream Stack Exchange Q&A from HuggingFaceH4/stack-exchange-preferences.
    Filters to records that contain code blocks; takes the highest-scored answer.
    Default max_records = 150K for experiment scale.
    """
    cap = max_records if max_records is not None else _SO_EXPERIMENT_MAX
    print(f"Streaming Stack Exchange Q&A (cap={cap:,}) ...")
    ds = load_dataset(
        "HuggingFaceH4/stack-exchange-preferences",
        split="train",
        streaming=True,
    )
    writer = ShardWriter(out_dir, "stackoverflow")
    kept = seen = 0

    for row in tqdm(ds, desc="stackoverflow"):
        seen += 1
        if cap and kept >= cap:
            break

        question = _strip_html(row.get("question") or "")
        answers   = row.get("answers") or []
        if not question or not answers:
            continue

        # Skip if no code in question or answers (filters out off-topic SE sites)
        all_text = question + " ".join(a.get("text", "") for a in answers)
        if not _has_code(all_text):
            continue

        # Pick the accepted answer, or else highest pm_score
        best = max(answers, key=lambda a: (a.get("selected", False), a.get("pm_score", 0)))
        answer = _strip_html(best.get("text") or "")
        if len(answer) < 100:
            continue

        content = f"Q: {question}\n\nA: {answer}"
        writer.write({"source": "stackoverflow", "content": content, "size": len(content)})
        kept += 1

    writer.close()
    print(f"  kept {kept:,} / {seen:,} ({100*kept/max(seen,1):.1f}%)")


# ---------------------------------------------------------------------------
# GitHub commits  (bigcode/commitpackft)
# ---------------------------------------------------------------------------

_COMMIT_LANGS    = {"Python", "Shell", "Bash"}
_TRIVIAL_COMMIT  = re.compile(r"^(merge|fix typo|update readme|bump version|wip|minor|cleanup)\.?$", re.IGNORECASE)
_COMMIT_EXPERIMENT_MAX = 200_000

def download_github_commits(out_dir: Path, max_records: int = None):
    """
    Stream Python + Shell commits from bigcode/commitpackft.
    Loads per-language JSONL files directly (bypasses the deprecated dataset script).
    Formats each record as the commit message followed by the resulting file.
    Default max_records = 200K for experiment scale.
    """
    cap = max_records if max_records is not None else _COMMIT_EXPERIMENT_MAX
    print(f"Streaming GitHub commits (cap={cap:,}) ...")

    # commitpackft has no Parquet — load per-language JSONL directly
    ds = load_dataset(
        "json",
        data_files={
            "train": [
                "hf://datasets/bigcode/commitpackft/data/python/data.jsonl",
                "hf://datasets/bigcode/commitpackft/data/shell/data.jsonl",
            ]
        },
        split="train",
        streaming=True,
    )
    writer = ShardWriter(out_dir, "github_commits")
    kept = seen = 0

    for row in tqdm(ds, desc="github_commits"):
        seen += 1
        if cap and kept >= cap:
            break

        lang = row.get("lang") or ""
        if lang not in _COMMIT_LANGS:
            continue

        subject     = (row.get("subject") or "").strip()
        new_contents = (row.get("new_contents") or "").strip()

        if not subject or not new_contents:
            continue
        if _TRIVIAL_COMMIT.match(subject):
            continue
        if len(new_contents) < 64:
            continue

        # Truncate large files — commit context, not full repo
        new_contents = new_contents[:MAX_FILE_BYTES]
        content = f"# {subject}\n\n{new_contents}"

        licenses = [row.get("license")] if row.get("license") else []
        writer.write({
            "source": "github_commits",
            "content": content,
            "lang": lang,
            "licenses": licenses,
            "size": len(content),
        })
        kept += 1

    writer.close()
    print(f"  kept {kept:,} / {seen:,} ({100*kept/max(seen,1):.1f}%)")


# ---------------------------------------------------------------------------
# GitHub issues  (bigcode/the-stack-github-issues)
# ---------------------------------------------------------------------------

# Strip bigcode's special delimiters; replace with clean separators
_ISSUE_TOKENS = re.compile(r"<issue_start>|<issue_comment>|<issue_closed>|<issue_opened>")
_ISSUES_EXPERIMENT_MAX = 150_000

def download_github_issues(out_dir: Path, max_records: int = None):
    """
    Stream GitHub issues from bigcode/the-stack-github-issues.
    Filters out PRs and very short threads; strips bigcode special tokens.
    Default max_records = 150K for experiment scale.

    NOTE: This dataset is gated. Accept the ToU at:
      https://huggingface.co/datasets/bigcode/the-stack-github-issues
    then ensure HF_TOKEN is set. A DatasetNotFoundError indicates missing access.
    """
    cap = max_records if max_records is not None else _ISSUES_EXPERIMENT_MAX
    print(f"Streaming GitHub issues (cap={cap:,}) ...")
    try:
        ds = load_dataset(
            "bigcode/the-stack-github-issues",
            split="train",
            streaming=True,
        )
    except Exception as e:
        if "gated" in str(e).lower() or "not found" in str(e).lower():
            print(f"""
ERROR: bigcode/the-stack-github-issues requires access approval.
  1. Visit https://huggingface.co/datasets/bigcode/the-stack-github-issues
  2. Accept the Terms of Use
  3. Re-run with HF_TOKEN set in .env
""")
            return
        raise
    writer = ShardWriter(out_dir, "github_issues")
    kept = seen = 0

    for row in tqdm(ds, desc="github_issues"):
        seen += 1
        if cap and kept >= cap:
            break

        # Skip pull requests
        if row.get("pull_request"):
            continue

        text_size = row.get("text_size") or 0
        if text_size < 200:
            continue

        content = row.get("content") or ""
        if not content:
            continue

        # Replace bigcode delimiters with readable separators
        content = _ISSUE_TOKENS.sub("\n---\n", content).strip()
        content = content[:MAX_FILE_BYTES]

        writer.write({
            "source": "github_issues",
            "content": content,
            "repo": row.get("repo") or "",
            "size": len(content),
        })
        kept += 1

    writer.close()
    print(f"  kept {kept:,} / {seen:,} ({100*kept/max(seen,1):.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and filter corpus sources")
    parser.add_argument("--out", required=True, help="Output directory for JSONL shards")
    ALL_SOURCES = [
        "stack_python", "stack_bash", "stack_md",
        "stackoverflow", "github_commits", "github_issues", "tldr",
    ]
    parser.add_argument("--source", choices=ALL_SOURCES, help="Single source to download")
    parser.add_argument("--all", action="store_true", help="Download all sources")
    parser.add_argument("--full", action="store_true", help="Full download (no caps); default is experiment scale")
    parser.add_argument("--so-xml", help="Path to Stack Overflow Posts.xml (legacy; HF source preferred)")
    parser.add_argument("--tldr-path", help="Path to cloned tldr-pages repo")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = ALL_SOURCES if args.all else ([args.source] if args.source else [])
    if not sources:
        parser.print_help()
        sys.exit(1)

    for source in sources:
        if source == "stack_python":
            download_stack(out_dir, "python", "stack_python", None,
                           None if args.full else _STACK_PYTHON_EXPERIMENT_MAX)
        elif source == "stack_bash":
            download_stack(out_dir, "shell", "stack_bash", None,
                           None if args.full else _STACK_BASH_EXPERIMENT_MAX)
        elif source == "stack_md":
            download_stack(out_dir, "markdown", "stack_md", None,
                           None if args.full else _STACK_MD_EXPERIMENT_MAX)
        elif source == "stackoverflow":
            if args.so_xml:
                download_stackoverflow(out_dir, args.so_xml)
            else:
                download_stackoverflow_hf(out_dir, None if args.full else _SO_EXPERIMENT_MAX)
        elif source == "github_commits":
            download_github_commits(out_dir, None if args.full else _COMMIT_EXPERIMENT_MAX)
        elif source == "github_issues":
            download_github_issues(out_dir, None if args.full else _ISSUES_EXPERIMENT_MAX)
        elif source == "tldr":
            download_tldr(out_dir, args.tldr_path)


if __name__ == "__main__":
    main()
