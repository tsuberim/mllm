"""
Fetch Python repos from GitHub one by one, scan for passing tests, keep only those that pass.

Strategy:
  - Search GitHub for Python repos with tests (pytest keyword in files)
  - Clone each repo
  - Run pytest in Docker sandbox
  - Keep if passing, delete immediately if not
  - Stop when target number of passing repos reached

Usage:
    python scripts/fetch_and_scan.py --target 50 --workers 4
    python scripts/fetch_and_scan.py --target 50 --workers 4 --stars-min 500 --stars-max 5000
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.sandbox import Sandbox, SANDBOX_IMAGE
from harness.task_gen import _install_deps, _run_pytest_in_sandbox

META    = Path("data/repos/meta.jsonl")
PASSING = Path("data/repos/passing.jsonl")
REPOS   = Path("data/repos")

GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def gh_search(page: int, query: str, already_seen: set) -> list[dict]:
    """Search GitHub for Python repos not already tried."""
    import urllib.parse
    encoded = urllib.parse.quote(query)
    cmd = [
        "gh", "api",
        f"search/repositories?q={encoded}&sort=stars&order=desc&per_page=50&page={page}",
        "--jq", '.items[] | {full_name, stargazers_count, size, license, description, default_branch, topics: (.topics // [])}',
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=60)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        if "rate limit" in stderr.lower() or "403" in str(e):
            time.sleep(60)
        return []
    except subprocess.TimeoutExpired:
        return []

    repos = []
    for line in out.strip().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["full_name"] in already_seen:
            continue
        lic = (r.get("license") or {})
        lic_key = lic.get("key", "") if isinstance(lic, dict) else ""
        if lic_key not in ("mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", ""):
            continue
        repos.append({
            "full_name": r["full_name"],
            "owner": r["full_name"].split("/")[0],
            "name": r["full_name"].split("/")[1],
            "stars": r["stargazers_count"],
            "size_kb": r["size"],
            "license": lic_key,
            "description": r.get("description", ""),
            "default_branch": r.get("default_branch", "main"),
            "topics": r.get("topics", []),
        })
    return repos


MIN_TEST_FILES = 5  # require at least 5 test_*.py files (proxy for ~20+ tests)


_rate_limit_until = [0.0]
_rate_limit_lock = threading.Lock()


def has_pytest(repo: dict) -> bool:
    """API check: repo has enough test files to be worth scanning."""
    # Respect shared rate-limit backoff
    with _rate_limit_lock:
        wait = _rate_limit_until[0] - time.monotonic()
    if wait > 0:
        time.sleep(wait)

    name = repo["full_name"]
    branch = repo["default_branch"]
    try:
        out = subprocess.check_output(
            ["gh", "api", f"repos/{name}/git/trees/{branch}?recursive=1",
             "--jq", '.tree[] | select(.type == "blob") | .path'],
            text=True, timeout=20,
        )
        paths = out.strip().splitlines()
        n = sum(
            1 for p in paths
            if p.endswith(".py") and (
                "/tests/" in p or p.startswith("tests/") or
                "/test/" in p or p.startswith("test/") or
                p.rsplit("/", 1)[-1].startswith("test_") or
                p.rsplit("/", 1)[-1].endswith("_test.py")
            )
        )
        return n >= MIN_TEST_FILES
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        if "rate limit" in stderr.lower() or "403" in str(e):
            with _rate_limit_lock:
                _rate_limit_until[0] = time.monotonic() + 60
        return False
    except Exception:
        return False


def clone_repo(repo: dict) -> Path | None:
    """Clone repo to data/repos/<slug>/. Returns path or None on failure."""
    slug = repo["full_name"].replace("/", "__")
    dest = REPOS / slug
    if dest.exists():
        return dest
    try:
        subprocess.check_call(
            ["git", "clone", "--depth=1", "--quiet",
             f"https://github.com/{repo['full_name']}.git", str(dest)],
            timeout=60,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return dest
    except Exception:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        return None


MIN_TESTS = 20  # require at least this many passing tests

# Packages that indicate tests need a running external service.
# Presence in requirements is a strong signal; we also scan test source directly.
_SERVICE_PACKAGES = {
    # databases
    "psycopg2", "psycopg2-binary", "psycopg", "asyncpg", "aiopg",
    "pymysql", "aiomysql", "mysqlclient",
    "sqlalchemy",           # often used with real DB URIs in tests
    "alembic",
    "pymongo", "motor",
    "redis", "aioredis",
    "elasticsearch", "opensearch-py",
    "cassandra-driver",
    "neo4j",
    "influxdb", "influxdb-client",
    # cloud / object storage
    "boto3", "botocore", "aiobotocore", "s3fs",
    "google-cloud-storage", "google-cloud-bigquery", "google-cloud-pubsub",
    "azure-storage-blob", "azure-servicebus",
    # message brokers
    "pika",                 # RabbitMQ
    "confluent-kafka", "kafka-python", "aiokafka",
    "celery",
    # service orchestration in tests
    "testcontainers",
    "pytest-docker",
    "docker",               # direct Docker SDK use in tests
    # email / comms
    "sendgrid", "twilio",
    # other network services
    "ldap3", "python-ldap",
    "paramiko",             # SSH
    "ftplib",               # stdlib but signals network use
    "smtplib",
}

# Patterns in test source that indicate live service connections.
# Uses simple substring / regex matching — fast, no AST needed.
_SERVICE_PATTERNS = [
    # DB connection strings / clients
    r"psycopg2\.connect\(",
    r"asyncpg\.connect\(",
    r"pymysql\.connect\(",
    r"pymongo\.MongoClient\(",
    r"motor\.motor_asyncio\.",
    r"redis\.Redis\(",
    r"redis\.StrictRedis\(",
    r"aioredis\.from_url\(",
    r"elasticsearch\.Elasticsearch\(",
    r"create_engine\(",          # SQLAlchemy — might be sqlite, checked below
    # docker / testcontainers
    r"testcontainers\.",
    r"DockerContainer\(",
    r"from docker import",
    r"import docker",
    # cloud SDKs
    r"boto3\.client\(",
    r"boto3\.resource\(",
    r"storage\.Client\(",
    # env-var guards that reveal integration test intent
    r'pytest\.skip\([^)]*(?:database|db|redis|mongo|rabbit|kafka|service)',
    r'os\.environ\[.(?:DATABASE_URL|REDIS_URL|MONGO_URI|BROKER_URL|S3_BUCKET)',
    r'os\.getenv\(.(?:DATABASE_URL|REDIS_URL|MONGO_URI|BROKER_URL)',
]

# SQLAlchemy with sqlite is fine — only flag real DB engines
_SQLITE_ONLY_RE = r'create_engine\(["\']sqlite'


def needs_external_services(repo_path: Path) -> bool:
    """
    Return True if the repo's tests likely require a running external service
    (database, Redis, S3, broker, etc.).

    Two checks (both after clone, no network):
    1. Service-indicating packages in any requirements manifest.
    2. Service-connection patterns in conftest.py and test source files.
    """
    import re

    # --- 1. Requirements manifests ---
    req_files = (
        list(repo_path.glob("requirements*.txt"))
        + list(repo_path.glob("*/requirements*.txt"))
        + list(repo_path.glob("**/requirements*.txt"))
    )
    for req_file in req_files:
        try:
            content = req_file.read_text(errors="replace").lower()
            for pkg in _SERVICE_PACKAGES:
                if pkg in content:
                    return True
        except Exception:
            pass

    # pyproject.toml — check [project] dependencies and [project.optional-dependencies]
    pyproject = repo_path / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text(errors="replace").lower()
            for pkg in _SERVICE_PACKAGES:
                if pkg in content:
                    return True
        except Exception:
            pass

    # --- 2. Source patterns in conftest.py and test files ---
    # Conftest first — service fixtures live there most often
    test_files: list[Path] = []
    for conftest in repo_path.rglob("conftest.py"):
        test_files.insert(0, conftest)     # prioritise conftests
    for tf in repo_path.rglob("test_*.py"):
        test_files.append(tf)
    for tf in repo_path.rglob("*_test.py"):
        test_files.append(tf)

    compiled = [re.compile(p, re.IGNORECASE) for p in _SERVICE_PATTERNS]
    sqlite_re = re.compile(_SQLITE_ONLY_RE, re.IGNORECASE)

    for path in test_files:
        try:
            source = path.read_text(errors="replace")
        except Exception:
            continue
        for pattern in compiled:
            if pattern.search(source):
                # create_engine with sqlite is acceptable
                if "create_engine" in pattern.pattern and sqlite_re.search(source):
                    continue
                return True

    return False


def check_repo(repo: dict, local_path: Path) -> tuple[bool, str]:
    """Returns (passes, status_message). Requires MIN_TESTS passing tests."""
    import re
    try:
        with Sandbox(repo_path=local_path, image=SANDBOX_IMAGE) as sb:
            _install_deps(sb, timeout=120)
            r = sb.bash(
                ".venv/bin/python -m pytest --tb=no -q --no-header --continue-on-collection-errors 2>&1",
                timeout=90,
            )
            failing = []
            n_passed = 0
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.startswith("FAILED "):
                    failing.append(line[len("FAILED "):].split(" - ")[0].strip())
                m = re.search(r"(\d+) passed", line)
                if m:
                    n_passed = int(m.group(1))

            if r.exit_code == 5 or (n_passed == 0 and not failing):
                return False, f"no tests ran"
            if failing:
                return False, f"fail ({len(failing)} failures)"
            if n_passed < MIN_TESTS:
                return False, f"too few tests ({n_passed} < {MIN_TESTS})"
            return True, f"PASS ({n_passed} tests)"
    except Exception as e:
        return False, f"err: {str(e)[:60]}"


def load_seen() -> set:
    seen = set()
    for path in (META, PASSING):
        if path.exists():
            with open(path) as f:
                for line in f:
                    try:
                        seen.add(json.loads(line)["full_name"])
                    except Exception:
                        pass
    return seen


def load_candidates(path: str) -> list[dict]:
    """Load a pre-built candidates JSONL (from search_repos.py)."""
    candidates = []
    with open(path) as f:
        for line in f:
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=50, help="Target number of passing repos")
    parser.add_argument("--workers", type=int, default=4, help="Parallel scan workers")
    parser.add_argument("--stars-min", type=int, default=200)
    parser.add_argument("--stars-max", type=int, default=10000)
    parser.add_argument("--candidates", help="Pre-built candidates JSONL from search_repos.py")
    parser.add_argument("--no-network-deps", action="store_true",
                        help="Skip repos whose tests need external services (DB, Redis, S3, etc.)")
    args = parser.parse_args()

    REPOS.mkdir(parents=True, exist_ok=True)

    already_seen = load_seen()
    n_passing = sum(1 for _ in open(PASSING)) if PASSING.exists() else 0
    print(f"Already passing: {n_passing} | Target: {args.target} | Need: {args.target - n_passing}")

    if n_passing >= args.target:
        print("Already at target.")
        return

    out_lock = threading.Lock()
    seen_lock = threading.Lock()
    candidate_q: list[dict] = []
    q_lock = threading.Lock()
    done = threading.Event()

    passing_count = [n_passing]

    if args.candidates:
        # Pre-built candidate list — load directly, no live search needed
        all_candidates = [
            r for r in load_candidates(args.candidates)
            if r["full_name"] not in already_seen
            and args.stars_min <= r.get("stars", 0) <= args.stars_max
        ]
        print(f"Loaded {len(all_candidates)} candidates from {args.candidates}")
        candidate_q.extend(all_candidates)
        fetcher = None
    else:
        def fetch_candidates():
            """Producer: rotates high-signal queries to find repos that actually use pytest."""
            queries = [
                f"topic:pytest language:python stars:{args.stars_min}..{args.stars_max} is:public archived:false",
                f"topic:testing language:python stars:{args.stars_min}..{args.stars_max} is:public archived:false",
                f"pytest in:readme language:python stars:{args.stars_min}..2000 is:public archived:false",
                f"pytest in:readme language:python stars:2001..{args.stars_max} is:public archived:false",
                f"topic:unit-testing language:python stars:{args.stars_min}..{args.stars_max} is:public archived:false",
            ]
            query_idx = 0
            page = 1
            while not done.is_set():
                query = queries[query_idx]
                with seen_lock:
                    seen_copy = set(already_seen)
                batch = gh_search(page, query, seen_copy)
                if not batch:
                    query_idx = (query_idx + 1) % len(queries)
                    page = 1
                    time.sleep(2)
                    continue
                with q_lock:
                    candidate_q.extend(batch)
                page += 1
                time.sleep(1)

        fetcher = threading.Thread(target=fetch_candidates, daemon=True)
        fetcher.start()
        time.sleep(2)  # let fetcher populate queue

    def worker():
        while not done.is_set():
            with q_lock:
                if not candidate_q:
                    if args.candidates:
                        return   # static list exhausted
                    time.sleep(0.5)
                    continue
                repo = candidate_q.pop(0)

            with seen_lock:
                if repo["full_name"] in already_seen:
                    continue
                already_seen.add(repo["full_name"])

            # Pre-check: skip repos with no test files.
            # has_conftest=True means search_repos.py already confirmed conftest.py exists.
            if not repo.get("has_conftest") and not has_pytest(repo):
                continue

            # Clone
            local_path = clone_repo(repo)
            if local_path is None:
                print(f"  CLONE FAIL  {repo['full_name']}", flush=True)
                continue

            # Service filter — cheap static scan, avoids Docker startup for rejected repos
            if args.no_network_deps and needs_external_services(local_path):
                shutil.rmtree(local_path, ignore_errors=True)
                print(f"  needs service                  {repo['full_name']}", flush=True)
                continue

            repo["local_path"] = str(local_path)

            with out_lock:
                with open(META, "a") as f:
                    f.write(json.dumps(repo) + "\n")

            passes, status = check_repo(repo, local_path)
            print(f"  {status:30s} {repo['full_name']}", flush=True)

            if passes:
                with out_lock:
                    with open(PASSING, "a") as f:
                        f.write(json.dumps(repo) + "\n")
                    passing_count[0] += 1
                    n = passing_count[0]
                print(f"  >>> {n}/{args.target} passing <<<", flush=True)
                if n >= args.target:
                    done.set()
            else:
                shutil.rmtree(local_path, ignore_errors=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\nDone. {passing_count[0]}/{args.target} repos passing → {PASSING}")


if __name__ == "__main__":
    main()
