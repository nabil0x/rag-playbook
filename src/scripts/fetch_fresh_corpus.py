#!/usr/bin/env python3
"""Fetch and verify the fresh benchmark corpora for projects P24-P36.

Downloads the verified-open sources below into `Data/corpus/<name>/` and logs
one line per file `url|sha256|bytes|relpath` to `Data/.corpus-manifest.txt`.

Sources (URLs re-verified 2026-08-04; gated/deprecated mirrors were dropped):

* `rag-mini-wikipedia`   - Wikipedia subset, P24-P26 core corpus (HF, CC BY 4.0)
* `beir-fiqa` / `beir-nfcorpus` - BEIR benchmarks incl. qrels (UKP Darmstadt
  server; the qrels live inside these zips - the old beir-cellar GitHub layout
  is gone)
* `lost-in-the-middle`   - nq-open gold-at-position subsets, P27 (GitHub
  `nelson-liu/lost-in-the-middle`; the HF namespace is not a dataset)
* `scifact`              - P28 attribution claims+corpus (official S3 release)
* `hotpotqa`             - P29-P31 dev distractor set (the CMU host is dead;
  served from the Wayback Machine snapshot)
* `mind-small`           - P35-P36 news recommendations (MSR research license,
  fetched only with `--mind`, mirrors the `--large` Enron gate)

* Idempotent: a file whose sha256 already matches the manifest is skipped;
  `--force` re-downloads it.
* First download is verified: magic bytes (PAR1 / PK / gzip / readable text)
  plus a minimum-size sanity check.
* Failure rule: primary URL fails -> try the alt URL; only then is the file
  marked FAILED in the manifest. Never silently pass a broken file.
* Zip/tar archives are extracted next to the archive after a clean download
  (manifest records the archive itself, matching the Enron handling in
  `fetch_sd_samples.py`).

Stdlib only (``requests`` optional).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.error
import urllib.request
from zipfile import ZipFile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "Data")
MANIFEST = os.path.join(DATA_DIR, ".corpus-manifest.txt")
USER_AGENT = "rag-playbook/1.0 contact@example.com"
HEX = set("0123456789abcdef")

# QREL_RELPATH = "qrels/test.tsv"  # (kept for reference: qrels are extracted from the zips)


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


def is_appledouble(name: str) -> bool:
    """True for macOS AppleDouble metadata members (._foo) shipped in some archives."""
    return os.path.basename(name).startswith("._")


def extract_archive(path: str, dest_dir: str) -> int:
    """Extract a zip or tar.gz next to the archive; returns member count.

    Rejects members that would escape *dest_dir* (ZipSlip / tar slip guard).
    """
    base = os.path.abspath(dest_dir)

    def safe(name: str) -> bool:
        target = os.path.abspath(os.path.join(dest_dir, name))
        return target == base or target.startswith(base + os.sep)

    if ZipFile.__module__ and os.path.exists(path):
        try:
            if not tarfile.is_tarfile(path):
                with ZipFile(path) as zf:
                    for m in zf.infolist():
                        if not safe(m.filename):
                            raise RuntimeError(f"unsafe zip member {m.filename!r}")
                    zf.extractall(dest_dir)
                    return len(zf.infolist())
        except (OSError, RuntimeError):
            pass

    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            members = [m for m in tf.getmembers() if (m.isfile() or m.isdir())
                       and not is_appledouble(m.name)]
            for m in members:
                if not safe(m.name):
                    raise RuntimeError(f"unsafe tar member {m.name!r}")
            kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
            tf.extractall(dest_dir, members=members, **kwargs)
            return len(members)
    return 0


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


# Corpus table — URLs re-verified 2026-08-04.
CORPUS = [
    # --- P24-P26 core corpus (CC BY 4.0) ---
    {"relpath": "corpus/rag-mini-wikipedia/passages.parquet", "primary": {"url": "https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia/resolve/main/data/passages.parquet/part.0.parquet", "magic": b"PAR1", "min_size": 100000}},
    {"relpath": "corpus/rag-mini-wikipedia/test.parquet", "primary": {"url": "https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia/resolve/main/data/test.parquet/part.0.parquet", "magic": b"PAR1", "min_size": 10000}},

    # --- P24-P26 BEIR benchmarks (corpus + queries + qrels inside the zip) ---
    {"relpath": "corpus/beir-fiqa/fiqa.zip", "primary": {"url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip", "magic": b"PK", "min_size": 1000000}, "extract": True},
    {"relpath": "corpus/beir-nfcorpus/nfcorpus.zip", "primary": {"url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip", "magic": b"PK", "min_size": 500000}, "extract": True},

    # --- P27 lost-in-the-middle (gold-at-position subsets) ---
    {"relpath": "corpus/lost-in-the-middle/nq-open-oracle.jsonl.gz", "primary": {"url": "https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/qa_data/nq-open-oracle.jsonl.gz", "magic": b"\x1f\x8b", "min_size": 100000}},
    {"relpath": "corpus/lost-in-the-middle/10_total_documents/nq-open-10_total_documents_gold_at_0.jsonl.gz", "primary": {"url": "https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/qa_data/10_total_documents/nq-open-10_total_documents_gold_at_0.jsonl.gz", "magic": b"\x1f\x8b", "min_size": 1000000}},
    {"relpath": "corpus/lost-in-the-middle/10_total_documents/nq-open-10_total_documents_gold_at_4.jsonl.gz", "primary": {"url": "https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/qa_data/10_total_documents/nq-open-10_total_documents_gold_at_4.jsonl.gz", "magic": b"\x1f\x8b", "min_size": 1000000}},
    {"relpath": "corpus/lost-in-the-middle/10_total_documents/nq-open-10_total_documents_gold_at_9.jsonl.gz", "primary": {"url": "https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/qa_data/10_total_documents/nq-open-10_total_documents_gold_at_9.jsonl.gz", "magic": b"\x1f\x8b", "min_size": 1000000}},
    {"relpath": "corpus/lost-in-the-middle/20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl.gz", "primary": {"url": "https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl.gz", "magic": b"\x1f\x8b", "min_size": 1000000}},
    {"relpath": "corpus/lost-in-the-middle/20_total_documents/nq-open-20_total_documents_gold_at_9.jsonl.gz", "primary": {"url": "https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_9.jsonl.gz", "magic": b"\x1f\x8b", "min_size": 1000000}},
    {"relpath": "corpus/lost-in-the-middle/20_total_documents/nq-open-20_total_documents_gold_at_19.jsonl.gz", "primary": {"url": "https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_19.jsonl.gz", "magic": b"\x1f\x8b", "min_size": 1000000}},

    # --- P28 SciFact (official release, CC BY-NC-SA 3.0) ---
    {"relpath": "corpus/scifact/data.tar.gz", "primary": {"url": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz", "magic": b"\x1f\x8b", "min_size": 1000000}, "extract": True},

    # --- P29-P31 HotpotQA dev distractor (CC BY-SA 4.0; CMU host dead -> Wayback) ---
    {"relpath": "corpus/hotpotqa/hotpot_dev_distractor_v1.json", "primary": {"url": "https://web.archive.org/web/2023id_/https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json", "magic": "text", "min_size": 10000000, "timeout": 600}, "alt": {"url": "https://web.archive.org/web/20230501000000id_/https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json", "magic": "text", "min_size": 10000000, "timeout": 600}},

    # --- P35-P36 MIND-small (MSR research license; --mind only) ---
    {"relpath": "corpus/mind-small/MIND_small_x1.zip", "opt": "mind", "primary": {"url": "https://huggingface.co/datasets/reczoo/MIND_small_x1/resolve/main/MIND_small_x1.zip", "magic": b"PK", "min_size": 1000000}, "extract": True},
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
        if spec.get("extract"):
            try:
                n = extract_archive(dest, os.path.dirname(dest))
                print(f"[extract] {n} members of {relpath} into {os.path.dirname(dest)}")
            except (OSError, RuntimeError) as e:
                print(f"[extract] WARNING {relpath}: {e} (archive kept, extraction skipped)")

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
    ap = argparse.ArgumentParser(description="Fetch and verify fresh benchmark corpora (P24-P36).")
    ap.add_argument("--force", action="store_true", help="re-download even if sha256 matches")
    ap.add_argument("--mind", action="store_true", help="also fetch MIND-small (MSR research license)")
    args = ap.parse_args()

    if args.mind:
        print("[license] MIND-small is released under the Microsoft Research License "
              "(https://msnews.github.io/) - non-commercial research use only.")

    manifest = load_manifest()
    lines, results = [], []
    for spec in CORPUS:
        if spec.get("opt") == "mind" and not args.mind:
            print(f"[skip] {spec['relpath']} (pass --mind to fetch)")
            continue
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
