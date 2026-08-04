#!/usr/bin/env python3
"""Fetch and verify three public-domain Project Gutenberg books.

Downloads the classic novels below into `Data/corpus/gutenberg/` and logs one
line per file `url|sha256|bytes|relpath` to `Data/.corpus-manifest.txt`.

Books (all public domain in the US; Project Gutenberg's standard UTF-8
plain-text cache editions are used):

* `pride-and-prejudice.txt`   - Pride and Prejudice, Jane Austen (1813)
* `moby-dick.txt`             - Moby Dick, Herman Melville (1851)
* `a-tale-of-two-cities.txt`  - A Tale of Two Cities, Charles Dickens (1859)

(URLs re-verified 2026-08-04; `gutenberg.org/cache/epub/` endpoints.)

* Idempotent: a file whose sha256 already matches the manifest is skipped and
  reported as "OK (cached)" - no network call at all; `--force` re-downloads.
* First download is verified: readable-text magic check plus a minimum-size
  sanity check (these are ~700 KB books, so `min_size` is 100000 bytes).
* Failure rule: a failed download is recorded as `FAILED` in the manifest so
  the next run retries it. Never silently pass a broken file.
* Manifest preservation: this script manages ONLY entries whose relpath starts
  with `corpus/gutenberg/`. Every other existing manifest entry (currently 13:
  rag-mini-wikipedia, beir-fiqa, beir-nfcorpus, lost-in-the-middle, scifact,
  hotpotqa) is carried over verbatim into the rewritten manifest, so running
  this fetcher never destroys the provenance of the other corpora.

Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Data")
MANIFEST = os.path.join(DATA_DIR, ".corpus-manifest.txt")
USER_AGENT = "rag-playbook/1.0 contact@example.com"
HEX = set("0123456789abcdef")

# Relpaths this script manages; everything else in the manifest is preserved.
GUTENBERG_RELPATH_PREFIX = "corpus/gutenberg/"


def _open(url, timeout=180, ua=None):
    """Open *url* with a descriptive User-Agent."""
    req = urllib.request.Request(url, headers={"User-Agent": ua or USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch(url: str, dest: str, timeout: int = 180, ua=None) -> bool:
    """Stream *url* to *dest*; True on success, False on any error."""
    try:
        with _open(url, timeout=timeout, ua=ua) as resp, open(dest, "wb") as fh:
            for chunk in iter(lambda: resp.read(65536), b""):
                fh.write(chunk)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return False


def is_text(data: bytes) -> bool:
    """Cheap plain-text test: no NUL bytes and mostly printable ASCII."""
    head = data[:8192]
    if b"\x00" in head:
        return False
    printable = sum(1 for b in head if b in (9, 10, 13) or 32 <= b < 127)
    return len(head) == 0 or printable / len(head) > 0.95


def verify(path: str, magic, min_size: int) -> bool:
    """Magic-byte + minimum-size sanity check for a downloaded file."""
    try:
        if os.path.getsize(path) < min_size:
            return False
        with open(path, "rb") as fh:
            head = fh.read(8)
        return is_text(head) if magic == "text" else head.startswith(magic)
    except OSError:
        return False


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    """Map relpath -> (url, sha256, bytes) from the previous manifest."""
    entries = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("|")
                if len(parts) == 4:
                    url, sha, size, relpath = parts
                    entries[relpath] = (url, sha, int(size) if size.isdigit() else 0)
    return entries


def save_manifest(lines: list) -> None:
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        fh.writelines(line + "\n" for line in sorted(lines))


# Gutenberg corpus table - URLs re-verified 2026-08-04 (public domain classics).
# Each book is a plain UTF-8 .txt (no archive extraction needed); gutenberg.org
# can be slow, hence the 240 s timeout.
GUTENBERG = [
    {"relpath": "corpus/gutenberg/pride-and-prejudice.txt", "primary": {"url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "magic": "text", "min_size": 100000, "timeout": 240}},
    {"relpath": "corpus/gutenberg/moby-dick.txt", "primary": {"url": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt", "magic": "text", "min_size": 100000, "timeout": 240}},
    {"relpath": "corpus/gutenberg/a-tale-of-two-cities.txt", "primary": {"url": "https://www.gutenberg.org/cache/epub/98/pg98.txt", "magic": "text", "min_size": 100000, "timeout": 240}},
]


def process_entry(spec: dict, manifest: dict, force: bool, lines: list, results: list) -> None:
    relpath = spec["relpath"]
    dest = os.path.join(DATA_DIR, relpath)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Idempotent skip: manifest sha matches the on-disk file.
    if not force and relpath in manifest and os.path.exists(dest):
        prev_url, prev_sha, _ = manifest[relpath]
        if len(prev_sha) == 64 and set(prev_sha) <= HEX and sha256_file(dest) == prev_sha:
            results.append((relpath, os.path.getsize(dest), prev_sha[:12], "OK (cached)"))
            lines.append(f"{prev_url}|{prev_sha}|{os.path.getsize(dest)}|{relpath}")
            return

    status, used_url, last_url = "FAILED", None, None
    for cand in [spec["primary"]] + ([spec["alt"]] if "alt" in spec else []):
        url = cand["url"]
        last_url = url
        if os.path.exists(dest):
            os.remove(dest)
        if fetch(url, dest, timeout=cand.get("timeout", 180)) and verify(dest, cand["magic"], cand["min_size"]):
            used_url, status = url, "OK" if cand is spec["primary"] else "fallback-used"
            break

    if status in ("OK", "fallback-used"):
        sha = sha256_file(dest)
        size = os.path.getsize(dest)
        lines.append(f"{used_url}|{sha}|{size}|{relpath}")
        results.append((relpath, size, sha[:12], status))
    else:
        # Never fabricate a file: record FAILED so the next run retries.
        lines.append(f"{last_url or spec['primary']['url']}|FAILED|0|{relpath}")
        results.append((relpath, 0, "-", "FAILED"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch and verify public-domain Project Gutenberg books.")
    ap.add_argument("--force", action="store_true", help="re-download even if sha256 matches")
    args = ap.parse_args()

    manifest = load_manifest()
    lines, results = [], []

    # Manifest preservation: `save_manifest()` rewrites the WHOLE manifest
    # file, so seed the outgoing lines with every existing entry whose
    # relpath is NOT managed by this script (anything outside
    # corpus/gutenberg/ - currently the 13 verified lines for
    # rag-mini-wikipedia, beir-fiqa, beir-nfcorpus, lost-in-the-middle,
    # scifact and hotpotqa), verbatim as stored. Without this seeding the
    # rewrite below would delete those entries.
    preserved = 0
    for relpath, (url, sha, size) in manifest.items():
        if not relpath.startswith(GUTENBERG_RELPATH_PREFIX):
            lines.append(f"{url}|{sha}|{size}|{relpath}")
            preserved += 1
    if preserved:
        print(f"[preserve] keeping {preserved} existing manifest entries not managed by this script")

    for spec in GUTENBERG:
        process_entry(spec, manifest, args.force, lines, results)
    save_manifest(lines)

    print(f"\n{'file':<55} {'size':>10}  {'sha256':<14} status")
    print("-" * 95)
    for relpath, size, sha, status in results:
        print(f"{relpath:<55} {size:>10,}  {sha:<14} {status}")

    def count(*statuses):
        return sum(1 for r in results if r[3] in statuses)

    failed = [r for r in results if r[3] == "FAILED"]
    print(f"\nSummary: {count('OK')} fetched, {count('OK (cached)')} cached (skipped), "
          f"{count('fallback-used')} fallback-used, {len(failed)} FAILED (of {len(results)})")
    for r in failed:
        print("  FAILED:", r[0])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
