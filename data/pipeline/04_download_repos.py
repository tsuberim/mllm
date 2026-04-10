"""
Download GitHub repos for use as agentic trace sandboxes.

Repos provide codebase diversity for trace generation — each trace runs inside
a real repo, so task variety follows naturally from repo variety.

Criteria: Python-primary, permissive license, 10–200 Python files, 10–5000 stars,
not a fork, has tests.

Output:
  data/repos/<owner>__<repo>/   — extracted repo contents
  data/repos/meta.jsonl         — one record per repo with metadata

Usage:
    python data/pipeline/04_download_repos.py --out data/repos/ --count 500
    python data/pipeline/04_download_repos.py --out data/repos/ --count 50 --dry-run
"""

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import time
from io import BytesIO
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # required for rate limits

MIN_STARS = 10
MAX_STARS = 5000        # avoid huge famous repos — keep it varied
MIN_PY_FILES = 5
MAX_PY_FILES = 200      # keep sandbox size manageable
MAX_REPO_MB = 50        # skip huge repos
PERMISSIVE_LICENSES = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause",
    "isc", "unlicense", "cc0-1.0", "wtfpl", "0bsd",
}

# Search queries to get variety in project types
SEARCH_QUERIES = [
    "language:Python topic:cli stars:10..5000",
    "language:Python topic:data-science stars:10..5000",
    "language:Python topic:web stars:10..5000",
    "language:Python topic:machine-learning stars:10..5000",
    "language:Python topic:automation stars:10..5000",
    "language:Python topic:tool stars:10..5000",
    "language:Python topic:library stars:10..5000",
    "language:Python topic:api stars:10..5000",
    "language:Python topic:testing stars:10..5000",
    "language:Python topic:devops stars:10..5000",
]

# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _get(url: str, params: dict = None) -> dict:
    r = requests.get(url, headers=_headers(), params=params, timeout=30)
    if r.status_code == 403 and "rate limit" in r.text.lower():
        reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
        wait = max(reset - time.time() + 2, 1)
        print(f"  rate limited — sleeping {wait:.0f}s")
        time.sleep(wait)
        return _get(url, params)
    r.raise_for_status()
    return r.json()


def search_repos(query: str, per_page: int = 100, max_results: int = 200) -> list[dict]:
    """Search GitHub repos, return list of repo metadata dicts."""
    results = []
    page = 1
    while len(results) < max_results:
        data = _get(f"{GITHUB_API}/search/repositories", params={
            "q": query + " fork:false",
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
            "page": page,
        })
        items = data.get("items", [])
        if not items:
            break
        results.extend(items)
        page += 1
        if len(items) < per_page:
            break
        time.sleep(0.5)  # be polite
    return results[:max_results]


def get_py_file_count(owner: str, repo: str) -> int:
    """
    Estimate Python file count from the repo languages API.
    Returns -1 on failure. Much faster than code search API.
    """
    try:
        data = _get(f"{GITHUB_API}/repos/{owner}/{repo}/languages")
        py_bytes = data.get("Python", 0)
        # Rough heuristic: avg Python file ~3KB
        return py_bytes // 3000
    except Exception:
        return -1


def download_tarball(owner: str, repo: str, ref: str, out_dir: Path) -> bool:
    """Download and extract default branch tarball into out_dir/<owner>__<repo>/."""
    url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{ref}.tar.gz"
    try:
        r = requests.get(url, headers=_headers(), timeout=60, stream=True)
        r.raise_for_status()
        content = b"".join(r.iter_content(chunk_size=65536))

        dest = out_dir / f"{owner}__{repo}"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)

        with tarfile.open(fileobj=BytesIO(content), mode="r:gz") as tar:
            # strip top-level dir (repo-branch/)
            members = tar.getmembers()
            if not members:
                return False
            prefix = members[0].name.split("/")[0] + "/"
            for member in members:
                member.name = member.name.removeprefix(prefix)
                if member.name:
                    tar.extract(member, dest, filter="data")

        return True
    except Exception as e:
        print(f"  failed to download {owner}/{repo}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def is_acceptable(repo: dict) -> bool:
    if repo.get("fork"):
        return False
    if repo.get("size", 0) > MAX_REPO_MB * 1024:
        return False
    if not MIN_STARS <= repo.get("stargazers_count", 0) <= MAX_STARS:
        return False
    license_key = (repo.get("license") or {}).get("spdx_id", "").lower()
    if license_key not in PERMISSIVE_LICENSES:
        return False
    return True


def run(out_dir: Path, target_count: int, dry_run: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.jsonl"

    # Load already-downloaded repos to avoid re-downloading
    already_done = set()
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                r = json.loads(line)
                already_done.add(r["full_name"])

    print(f"Already downloaded: {len(already_done)} repos. Target: {target_count}")

    candidates = []
    per_query = max(target_count // len(SEARCH_QUERIES) * 3, 50)  # oversample for filtering

    print("Searching GitHub...")
    for query in SEARCH_QUERIES:
        results = search_repos(query, max_results=per_query)
        for r in results:
            if r["full_name"] not in already_done:
                candidates.append(r)

    # Deduplicate by full_name
    seen = set()
    unique = []
    for r in candidates:
        if r["full_name"] not in seen:
            seen.add(r["full_name"])
            unique.append(r)

    # Filter basic criteria
    filtered = [r for r in unique if is_acceptable(r)]
    print(f"Candidates after basic filter: {len(filtered)}")

    downloaded = 0
    with open(meta_path, "a") as meta_f:
        for repo in tqdm(filtered, desc="repos"):
            if downloaded >= target_count - len(already_done):
                break

            owner = repo["owner"]["login"]
            name = repo["name"]
            full_name = repo["full_name"]
            default_branch = repo.get("default_branch", "main")

            # Check Python file count via API
            py_count = get_py_file_count(owner, name)
            if not (MIN_PY_FILES <= py_count <= MAX_PY_FILES):
                continue

            if dry_run:
                print(f"  [dry-run] would download {full_name} ({py_count} py files, {repo['stargazers_count']} stars)")
                downloaded += 1
                continue

            success = download_tarball(owner, name, default_branch, out_dir)
            if not success:
                continue

            meta = {
                "full_name": full_name,
                "owner": owner,
                "name": name,
                "stars": repo["stargazers_count"],
                "py_files": py_count,
                "size_kb": repo["size"],
                "license": (repo.get("license") or {}).get("spdx_id", "unknown"),
                "topics": repo.get("topics", []),
                "description": repo.get("description", ""),
                "default_branch": default_branch,
                "local_path": str(out_dir / f"{owner}__{name}"),
            }
            meta_f.write(json.dumps(meta) + "\n")
            meta_f.flush()
            downloaded += 1
            time.sleep(0.2)

    print(f"\nDone. Downloaded {downloaded} new repos → {out_dir}")
    print(f"Total in meta.jsonl: {len(already_done) + downloaded}")


def main():
    parser = argparse.ArgumentParser(description="Download GitHub repos for sandbox diversity")
    parser.add_argument("--out", required=True, help="Output directory for repos")
    parser.add_argument("--count", type=int, default=500, help="Number of repos to download")
    parser.add_argument("--dry-run", action="store_true", help="List candidates without downloading")
    args = parser.parse_args()

    if not GITHUB_TOKEN and not args.dry_run:
        print("Warning: GITHUB_TOKEN not set. Rate limits will be very low (60 req/hr).")
        print("Set GITHUB_TOKEN env var for 5000 req/hr.\n")

    run(Path(args.out), args.count, args.dry_run)


if __name__ == "__main__":
    main()
