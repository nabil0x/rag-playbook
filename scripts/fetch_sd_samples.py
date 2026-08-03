#!/usr/bin/env python3
"""Fetch and verify the Special Documents (SD) sample files.

Downloads the committed samples from `.omo/plans/special-documents-phase.md`
section 4.2 into `Data/SD-0N-<slug>/` and logs one line per file
`url|sha256|bytes|relpath` to `Data/.samples-manifest.txt`.

* Idempotent: a file whose sha256 already matches the manifest is skipped;
  `--force` re-downloads it.
* First download is verified: magic bytes (%PDF / PK / readable text) plus a
  minimum-size sanity check against section 4.2.
* Failure rule (section 4.1): primary URL fails -> try the alt URL -> try the
  stdlib generator (SampleSS.xlsx per §4.3, .eml via stdlib email); only then
  is the sample marked FAILED in the manifest. Never silently pass a broken file.
* `--large` fetches the Enron corpus into `Data/SD-04-email/enron/`
  (gitignored). Not run by default.

Stdlib only (``requests`` optional).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Data")
MANIFEST = os.path.join(DATA_DIR, ".samples-manifest.txt")
USER_AGENT = "rag-playbook/1.0 contact@example.com"
ENRON_URL = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz"
HEX = set("0123456789abcdef")


def _open(url, timeout=180):
    """Open *url* with a descriptive User-Agent (SEC requires one)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def ia_file_url(item_id: str, needle: str) -> str:
    """archive.org download URL using the EXACT file `name` from the metadata API.

    The naive ``{id}_jp2.zip`` pattern 404s for DLI items, so we read
    ``https://archive.org/metadata/{id}`` and URL-encode the matched name.
    """
    with _open(f"https://archive.org/metadata/{item_id}") as resp:
        data = json.loads(resp.read().decode("utf-8"))
    match = next((f["name"] for f in data.get("files", []) if needle in f["name"]), None)
    if match is None:
        raise RuntimeError(f"no file matching {needle!r} in archive.org item {item_id!r}")
    return "https://archive.org/download/{}/{}".format(item_id, urllib.parse.quote(match))


def candidate_url(cand: dict) -> str:
    """Resolve a candidate: static `url` or archive.org `ia` metadata lookup."""
    return ia_file_url(cand["ia"]["item"], cand["ia"]["needle"]) if "ia" in cand else cand["url"]


def fetch(url: str, dest: str) -> bool:
    """Stream *url* to *dest*; True on success, False on any error."""
    try:
        with _open(url) as resp, open(dest, "wb") as fh:
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


def gen_sample_xlsx(path: str) -> bool:
    """Generate `SampleSS.xlsx` (plan §4.3): openpyxl if present, else stdlib ZIP/OOXML."""
    sheets = [
        ("Sales", [("date", "product", "units", "unit_price", "total"),
                   ("2026-01-05", "Widget A", 3, 10.0, 30.0), ("2026-01-12", "Gadget B", 5, 24.5, 122.5),
                   ("2026-02-03", "Widget A", 2, 10.0, 20.0), ("2026-02-17", "Gizmo C", 4, 8.75, 35.0),
                   ("2026-03-01", "Gadget B", 6, 24.5, 147.0), ("2026-03-22", "Gizmo C", 1, 8.75, 8.75)]),
        ("Employees", [("id", "name", "department", "joining_date"),
                       (101, "Amina Rahman", "Sales", "2023-04-11"), (102, "Rafiq Ahmed", "Engineering", "2022-09-01"),
                       (103, "Sadia Islam", "Finance", "2024-01-15"), (104, "Tanvir Hasan", "Operations", "2021-11-30")]),
        ("Products", [("sku", "name", "category", "stock"),
                      ("SKU-1001", "Widget A", "Hardware", 250), ("SKU-1002", "Gadget B", "Electronics", 80),
                      ("SKU-1003", "Gizmo C", "Accessories", 340)]),
    ]
    try:
        import openpyxl

        wb = openpyxl.Workbook()
        for i, (name, rows) in enumerate(sheets):
            ws = wb.active if i == 0 else wb.create_sheet()
            ws.title = name
            for row in rows:
                ws.append(row)
        wb.save(path)
        return True
    except ImportError:
        pass

    def col(n):
        s = ""
        while n:
            n, r = divmod(n - 1, 26)
            s = chr(65 + r) + s
        return s

    def cell(ref, v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return f'<c r="{ref}"><v>{v}</v></c>'
        t = str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<c r="{ref}" t="inlineStr"><is><t>{t}</t></is></c>'

    def sheet_xml(rows):
        body = "".join(
            f'<row r="{i}">' + "".join(cell(f"{col(j)}{i}", v) for j, v in enumerate(row, 1)) + "</row>"
            for i, row in enumerate(rows, 1)
        )
        return f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{W}"><sheetData>{body}</sheetData></worksheet>'

    S, W, R, DOC = (f"http://schemas.openxmlformats.org/{p}" for p in
                    ("package/2006/content-types", "spreadsheetml/2006/main",
                     "package/2006/relationships", "officeDocument/2006/relationships"))
    xml = lambda s: f'<?xml version="1.0" encoding="UTF-8"?>{s}'
    try:
        parts = {
            "[Content_Types].xml": xml(
                f'<Types xmlns="{S}">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                + "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, 4))
                + "</Types>"),
            "_rels/.rels": xml(
                f'<Relationships xmlns="{R}">'
                f'<Relationship Id="rId1" Type="{DOC}/officeDocument" Target="xl/workbook.xml"/>'
                "</Relationships>"),
            "xl/workbook.xml": xml(
                f'<workbook xmlns="{W}" xmlns:r="{DOC}"><sheets>'
                + "".join(f'<sheet name="{n}" sheetId="{i}" r:id="rId{i}"/>' for i, (n, _) in enumerate(sheets, 1))
                + "</sheets></workbook>"),
            "xl/_rels/workbook.xml.rels": xml(
                f'<Relationships xmlns="{R}">'
                + "".join(f'<Relationship Id="rId{i}" Type="{DOC}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, 4))
                + "</Relationships>"),
        }
        for i, (_, rows) in enumerate(sheets, 1):
            parts[f"xl/worksheets/sheet{i}.xml"] = sheet_xml(rows)
        with ZipFile(path, "w", ZIP_DEFLATED) as zf:
            for name, content in parts.items():
                zf.writestr(name, content)
        return True
    except OSError:
        return False


def gen_sample_eml(path: str) -> bool:
    """Generate a nested-attachment .eml with the stdlib `email` package."""
    try:
        outer = MIMEMultipart("mixed")
        outer["From"], outer["To"] = "alice@example.com", "bob@example.com"
        outer["Subject"], outer["Date"] = "Proposal with nested attachment", formatdate(usegmt=True)
        outer.attach(MIMEText("Hi Bob,\n\nHere is the latest proposal. The original "
                              "draft is attached below the memo.\n\nAlice\n"))
        inner = MIMEMultipart("mixed")
        inner.attach(MIMEText("Original draft (v1)\n----------------------\nBudget for Q3:\n"
                              "- research: 1200\n- tooling: 850\n- travel: 400\n\nSign off: Alice\n"))
        inner.attach(MIMEText("budget-v1.txt", "plain", "utf-8"))
        outer.attach(inner)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(outer.as_string())
        return True
    except OSError:
        return False


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


# Sample table — plan section 4.2, URLs re-verified 2026-08-03.
SAMPLES = [
    {"relpath": "SD-01-word/blk-paras-and-tables.docx", "primary": {"url": "https://raw.githubusercontent.com/python-openxml/python-docx/master/features/steps/test_files/blk-paras-and-tables.docx", "magic": b"PK", "min_size": 5000}, "alt": {"url": "https://raw.githubusercontent.com/python-openxml/python-docx/master/features/steps/test_files/tbl-having-tables.docx", "magic": b"PK", "min_size": 5000}},
    {"relpath": "SD-02-ppt/prs-notes.pptx", "primary": {"url": "https://raw.githubusercontent.com/scanny/python-pptx/master/features/steps/test_files/prs-notes.pptx", "magic": b"PK", "min_size": 30000}, "alt": {"url": "https://raw.githubusercontent.com/scanny/python-pptx/master/features/steps/test_files/cht-charts.pptx", "magic": b"PK", "min_size": 30000}},
    {"relpath": "SD-03-excel/SampleSS.xlsx", "primary": {"url": "https://raw.githubusercontent.com/apache/poi/trunk/test-data/spreadsheet/SampleSS.xlsx", "magic": b"PK", "min_size": 2000}, "gen": gen_sample_xlsx},
    {"relpath": "SD-04-email/raw_email_with_nested_attachment.eml", "primary": {"url": "https://raw.githubusercontent.com/mikel/mail/master/spec/fixtures/emails/mime_emails/raw_email_with_nested_attachment.eml", "magic": "text", "min_size": 1000}, "gen": gen_sample_eml},
    {"relpath": "SD-05-ocr/gilman1892.pdf", "primary": {"ia": {"item": "gilman1892", "needle": "gilman1892.pdf"}, "magic": b"%PDF", "min_size": 1000000}, "alt": {"ia": {"item": "yellowwallpaper00gilmgoog", "needle": "yellowwallpaper00gilmgoog.pdf"}, "magic": b"%PDF", "min_size": 50000}},
    {"relpath": "SD-05-ocr/gitanjali1914_jp2.zip", "primary": {"ia": {"item": "Gitanjali1914RabindranathTagore", "needle": "_jp2.zip"}, "magic": b"PK", "min_size": 5000000}, "alt": {"ia": {"item": "in.ernet.dli.2015.357160", "needle": "2015.357160.Gitanjali-_jp2.zip"}, "magic": b"PK", "min_size": 5000000}},
    {"relpath": "SD-06-tables/f1040.pdf", "primary": {"url": "https://www.irs.gov/pub/irs-pdf/f1040.pdf", "magic": b"%PDF", "min_size": 100000}, "alt": {"url": "https://www.sec.gov/Archives/edgar/data/0000320193/000032019323000106/aapl-20230930.htm", "relpath": "SD-06-tables/aapl-20230930.htm", "magic": "text", "min_size": 500000}},
    {"relpath": "SD-06-tables/aapl-20230930.htm", "primary": {"url": "https://www.sec.gov/Archives/edgar/data/0000320193/000032019323000106/aapl-20230930.htm", "magic": "text", "min_size": 500000}},
    {"relpath": "SD-07-chat/sample-chat.txt", "primary": {"url": "https://raw.githubusercontent.com/mutluksap/whatsapp-chat-export-viewer/main/docs/sample-chat.txt", "magic": "text", "min_size": 200}, "alt": {"url": "https://raw.githubusercontent.com/roboteam-digital/telegram-json-ui/main/frontend/sveltekit/static/example/telegram-test-json/result.json", "relpath": "SD-07-chat/result.json", "magic": "text", "min_size": 200}},
    {"relpath": "SD-08-invoices/Invoice_1.pdf", "primary": {"url": "https://raw.githubusercontent.com/Azure-Samples/azure-openai-gpt-4-vision-pdf-extraction-sample/main/Invoice_1.pdf", "magic": b"%PDF", "min_size": 100000}, "alt": {"url": "https://raw.githubusercontent.com/Azure/azure-sdk-for-js/main/sdk/formrecognizer/ai-form-recognizer/assets/invoice/Invoice_1.pdf", "magic": b"%PDF", "min_size": 100000}},
]


def process_sample(spec: dict, manifest: dict, force: bool, lines: list, results: list) -> None:
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
        try:
            url = candidate_url(cand)
        except (RuntimeError, OSError, ValueError):
            continue
        last_url = url
        if os.path.exists(dest):
            os.remove(dest)
        if fetch(url, dest) and verify(dest, cand["magic"], cand["min_size"]):
            used_url, status = url, "OK" if cand is spec["primary"] else "fallback-used"
            break

    # Stdlib generator fallback (SampleSS.xlsx per §4.3 / raw .eml).
    if status != "OK" and "gen" in spec:
        if os.path.exists(dest):
            os.remove(dest)
        if spec["gen"](dest) and verify(dest, spec["primary"]["magic"], spec["primary"]["min_size"]):
            used_url, status = "generated-in-script", "generated"

    if status in ("OK", "fallback-used", "generated"):
        sha = sha256_file(dest)
        size = os.path.getsize(dest)
        lines.append(f"{used_url}|{sha}|{size}|{relpath}")
        results.append((relpath, size, sha[:12], status))
    else:
        # Never fabricate a file: record FAILED so the next run retries.
        lines.append(f"{last_url or spec['primary'].get('url', 'n/a')}|FAILED|0|{relpath}")
        results.append((relpath, 0, "-", "FAILED"))


def fetch_enron(force: bool) -> None:
    """--large only: download + extract the Enron corpus (gitignored dir)."""
    target = os.path.join(DATA_DIR, "SD-04-email", "enron")
    os.makedirs(target, exist_ok=True)
    tar_path = os.path.join(target, "enron_mail_20150507.tar.gz")
    if force or not os.path.exists(tar_path):
        print(f"[enron] downloading {ENRON_URL}")
        if not fetch(ENRON_URL, tar_path):
            print("[enron] FAILED to download Enron corpus")
            return
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and "/allen-p/inbox/" in m.name and m.name.endswith(".eml")]
        tf.extractall(target, members=members)
    print(f"[enron] extracted {len(members)} .eml files into {target}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch and verify Special Documents samples.")
    ap.add_argument("--force", action="store_true", help="re-download even if sha256 matches")
    ap.add_argument("--large", action="store_true", help="fetch Enron corpus (147 MB, gitignored)")
    args = ap.parse_args()

    if args.large:
        fetch_enron(args.force)
        return 0

    manifest = load_manifest()
    lines, results = [], []
    for spec in SAMPLES:
        process_sample(spec, manifest, args.force, lines, results)
    save_manifest(lines)

    print(f"\n{'file':<45} {'size':>10}  {'sha256':<14} status")
    print("-" * 85)
    for relpath, size, sha, status in results:
        print(f"{relpath:<45} {size:>10,}  {sha:<14} {status}")

    def count(*statuses):
        return sum(1 for r in results if r[3] in statuses)

    failed = [r for r in results if r[3] == "FAILED"]
    print(f"\nSummary: {count('OK')} fetched, {count('OK (cached)')} cached (skipped), "
          f"{count('fallback-used', 'generated')} fallback-used/generated, "
          f"{len(failed)} FAILED (of {len(results)})")
    for r in failed:
        print("  FAILED:", r[0])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
