#!/usr/bin/env python3
"""Generate 5 synthetic Bangladeshi sample documents + ground-truth schema.json files.

Implements phase §7 of `.omo/plans/special-documents-phase.md`:

  1. bKash statement PDF            -> Data/SD-06-tables/bkash_statement.pdf
  2. Nagad statement PDF            -> Data/SD-06-tables/nagad_statement.pdf
  3. Mushak 6.3 VAT invoice PDF     -> Data/SD-08-invoices/mushak63_invoice.pdf
  4. Bilingual restaurant bill PDF  -> Data/SD-08-invoices/bilingual_restaurant_bill.pdf
  5. Bengali WhatsApp chat txt      -> Data/SD-07-chat/bangla_chat.txt

Every `<name>.schema.json` companion is written from the SAME in-memory data
structures used to render the document (single source of truth), so QA can
verify retrieval against exact ground-truth values.

Design notes
------------
* Deterministic: a fixed default seed (42) drives all randomness; `--seed N`
  overrides. Per-document generators get a stable integer sub-seed derived
  from the CLI seed (no reliance on Python's salted builtin `hash()`), so two
  runs on the same machine produce byte-identical output.
* Money is handled with `decimal.Decimal` (quantized to 0.01) and formatted
  with western thousands separators, e.g. ``1,250.00``.
* Bengali text is rendered with ONE Bengali font (Noto Sans Bengali), shaped
  with HarfBuzz (reportlab >= 5.0 auto-shapes TTF fonts when the optional
  `uharfbuzz` package is importable). A script-run splitting helper wraps
  Bengali-block runs in `<font name="BN">…</font>` and leaves Latin/ASCII runs
  in a Latin font, because reportlab's shaped (HarfBuzz) frag words lose the
  ToUnicode mapping of non-Bengali glyphs. Bengali glyphs therefore render
  correctly AND survive PDF text extraction.

Usage
-----
    python3 src/scripts/generate_bd_samples.py [--seed 42]

Requirements: reportlab (>= 4), pypdf or pdfplumber (verification), and a
Unicode TTF with Bengali coverage (auto-discovered, with a download fallback).
"""

import argparse
import html
import json
import random
import re
import string
import subprocess
import sys
import urllib.request
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# --------------------------------------------------------------------------
# Paths & font discovery
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Data"
SD06 = DATA_DIR / "SD-06-tables"
SD07 = DATA_DIR / "SD-07-chat"
SD08 = DATA_DIR / "SD-08-invoices"
FONT_DIR = REPO_ROOT / "src" / "scripts" / "fonts"

BENGALI_BLOCK = range(0x0980, 0x0A00)  # includes ৳ (U+09F3)

BENGALI_FONT_CANDIDATES = [
    "NotoSansBengali-Regular.ttf",  # under src/scripts/fonts/
    "/usr/share/fonts/noto/NotoSansBengali-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
    "/usr/share/fonts/google/noto-sans-bengali/NotoSansBengali-Regular.ttf",
    "/usr/share/fonts/truetype/NotoSansBengali-Regular.ttf",
    "/usr/share/fonts/TTF/NotoSansBengali-Regular.ttf",
    "/usr/share/fonts/wps-fonts/Nirmala.ttf",  # has Bengali coverage
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
]
BENGALI_BOLD_CANDIDATES = [
    "NotoSansBengali-Bold.ttf",
    "/usr/share/fonts/noto/NotoSansBengali-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansBengali-Bold.ttf",
    "/usr/share/fonts/google/noto-sans-bengali/NotoSansBengali-Bold.ttf",
]
LATIN_FONT_CANDIDATES = [
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]
LATIN_BOLD_CANDIDATES = [
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# Google Fonts (github.com/google/fonts) fallback URLs for the Bengali font.
FONT_DOWNLOAD_URLS = [
    "https://github.com/google/fonts/raw/main/ofl/notosansbengali/NotoSansBengali%5Bwdth%2Cwght%5D.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansbengali/NotoSansBengali%5Bwdth%2Cwght%5D.ttf",
    "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansBengali/hinted/ttf/NotoSansBengali-Regular.ttf",
]


def _fc_list_bangla():
    """Return font file paths that fc-list claims cover Bengali."""
    try:
        out = subprocess.run(
            ["fc-list", ":lang=bn", "file"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return [line.split(":", 1)[0].strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def find_bengali_font():
    """Locate a TTF with Bengali coverage; download into src/scripts/fonts/ as fallback."""
    for cand in BENGALI_FONT_CANDIDATES:
        p = cand if cand.startswith("/") else str(FONT_DIR / cand)
        if Path(p).is_file():
            return p
    for p in _fc_list_bangla():
        if p.lower().endswith((".ttf", ".otf")) and "noto" in p.lower() and "serif" not in p.lower():
            return p
    for p in _fc_list_bangla():
        if p.lower().endswith((".ttf", ".otf")):
            return p
    # Download fallback into src/scripts/fonts/ (single regular TTF, <= ~1 MB).
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    dest = FONT_DIR / "NotoSansBengali-Regular.ttf"
    for url in FONT_DOWNLOAD_URLS:
        try:
            print(f"  [font] downloading {url}", file=sys.stderr)
            urllib.request.urlretrieve(url, dest)
            if dest.stat().st_size < 1_000_000:
                print(f"  [font] saved {dest} ({dest.stat().st_size} bytes)", file=sys.stderr)
                return str(dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  [font] download failed ({url}): {exc}", file=sys.stderr)
    raise SystemExit(
        "No usable Bengali font found and automatic download failed. "
        "Please place NotoSansBengali-Regular.ttf in src/scripts/fonts/."
    )


def find_latin_fonts():
    for cand in LATIN_FONT_CANDIDATES:
        if Path(cand).is_file():
            bold = next((c for c in LATIN_BOLD_CANDIDATES if Path(c).is_file()), cand)
            return cand, bold
    return "Helvetica", "Helvetica-Bold"  # reportlab built-in Type1 fallback


def is_bengali(ch):
    return ord(ch) in BENGALI_BLOCK


def script_runs(text):
    """Yield (is_bengali_run, chunk) runs, splitting a string by script."""
    out, cur, cur_flag = [], "", None
    for ch in str(text):
        flag = is_bengali(ch)
        if flag != cur_flag:
            if cur:
                out.append((cur_flag, cur))
            cur, cur_flag = ch, flag
        else:
            cur += ch
    if cur:
        out.append((cur_flag, cur))
    return out


def markup(text, bn="BN", lat="LAT"):
    """Build reportlab Paragraph markup, tagging Bengali runs with the BN font."""
    parts = []
    for flag, chunk in script_runs(str(text)):
        tag = bn if flag else lat
        parts.append(f'<font name="{tag}">{html.escape(chunk)}</font>')
    return "".join(parts)


# --------------------------------------------------------------------------
# Randomness & money helpers
# --------------------------------------------------------------------------

ALNUM = string.ascii_uppercase + string.digits
DIGITS = string.digits


def subseed(seed, name):
    """Stable, process-independent integer sub-seed for a document generator."""
    s = int(seed) & 0xFFFFFFFF
    for b in name.encode("utf-8"):
        s = ((s * 33) ^ b) & 0xFFFFFFFF
    return s


def q2(v):
    return Decimal(v).quantize(Decimal("0.01"))


def fmt2(v):
    """1,250.00 - western thousands separators, 2 decimals."""
    return f"{Decimal(v):,.2f}"


def phone(rng):
    return "01" + rng.choice("35789") + "".join(rng.choices(DIGITS, k=8))


def trx_id(rng, prefix):
    return prefix + "".join(rng.choices(ALNUM, k=8))


# --------------------------------------------------------------------------
# Font registration
# --------------------------------------------------------------------------

def register_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    bn_path = find_bengali_font()
    lat_path, lat_bold_path = find_latin_fonts()

    pdfmetrics.registerFont(TTFont("BN", bn_path))  # shapable defaults to True

    bnb_path = None
    for c in BENGALI_BOLD_CANDIDATES:
        p = Path(c) if c.startswith("/") else FONT_DIR / c
        if p.is_file():
            bnb_path = str(p)
            break
    pdfmetrics.registerFont(TTFont("BNB", bnb_path or bn_path))

    if lat_path != "Helvetica":
        pdfmetrics.registerFont(TTFont("LAT", lat_path))
        pdfmetrics.registerFont(TTFont("LATB", lat_bold_path))
    else:
        # Built-in Type1 fonts are already registered as Helvetica/Helvetica-Bold.
        pdfmetrics.registerFont(TTFont("LAT", "Helvetica"))
        pdfmetrics.registerFont(TTFont("LATB", "Helvetica-Bold"))

    return {"bengali": bn_path, "latin": lat_path}

# --------------------------------------------------------------------------
# bKash statement (SD-06)
# --------------------------------------------------------------------------

BKASH_TYPES = [
    "Cash In", "Cash Out", "Send Money", "Mobile Recharge", "Make Payment",
    "Pay Bill", "Cashback", "Disbursement Received", "bKash to Bank",
    "Bank to bKash", "Remittance Received",
]
BKASH_TRX_PREFIX = {
    "Cash In": "CI", "Cash Out": "CO", "Send Money": "SM",
    "Mobile Recharge": "MR", "Make Payment": "MP", "Pay Bill": "PB",
    "Cashback": "CB", "Disbursement Received": "DB", "bKash to Bank": "BB",
    "Bank to bKash": "BT", "Remittance Received": "RM",
}
BKASH_RANGES = {
    "Cash In": (500, 25000, 100),
    "Cash Out": (500, 20000, 100),
    "Send Money": (100, 3000, 50),
    "Mobile Recharge": (50, 1000, 10),
    "Make Payment": (200, 8000, 50),
    "Pay Bill": (300, 15000, 100),
    "Cashback": (20, 500, 5),
    "Disbursement Received": (1000, 15000, 500),
    "bKash to Bank": (5000, 50000, 500),
    "Bank to bKash": (5000, 40000, 500),
    "Remittance Received": (5000, 100000, 500),
}
BKASH_WEIGHTS = [
    ("Cash In", 3), ("Cash Out", 4), ("Send Money", 4), ("Mobile Recharge", 2),
    ("Make Payment", 3), ("Pay Bill", 2), ("Cashback", 1),
    ("Disbursement Received", 1), ("bKash to Bank", 1), ("Bank to bKash", 1),
    ("Remittance Received", 1),
]
BKASH_MERCHANTS = [
    "Shwapno - Gulshan Avenue", "Khaja Ghar Restaurant, Dhanmondi",
    "Daraz Bangladesh", "Uber Bangladesh", "Bashundhara City Shopping",
    "Aarong - Tejgaon", "Chillox Burger - Banani", "Kacchi Bhai - Mohammadpur",
    "Agora Super Shop - Dhanmondi", "Meena Bazar - Mirpur 10",
    "Pran-RFL Group", "Bikroy.com", "Pathao - Ride Share", "Sheba.xyz",
    "Foodpanda Bangladesh", "WellFood - Uttara", "Sajida Fashion",
    "Dangdang BD - Shopping", "Evaly - Online Mart", "Star Kabab - Nilkhet",
    "Sultana Sweets - Gawsia", "Haji Biriyani - Dhaka", "Arambagh Super Market",
    "New Market Clothing", "Green Super Shop - Jatrabari",
]
BKASH_CASH_OUT_AGENTS = [
    "Cash Out Agent - Gulshan 1", "bKash Agent - Mirpur 11",
    "Agent Banking - Rupnagar", "Nogor Mela Agent - Uttara Sector 7",
    "Dcash Agent - Motijheel", "Agent Point - Badda",
    "Cash Out Agent - Banani 11", "bKash Agent - Mohammadpur",
    "Mobile Banker - Shyamoli", "Agent - Khilgaon Taltola",
]
BKASH_CASH_IN_AGENTS = [
    "Cash In via Agent - Badda", "Cash In Agent - Uttara",
    "Cash In via Agent - Motijheel", "Cash In - Agent Point Gulshan",
    "Cash In via Agent - Dhanmondi",
]
BKASH_SEND_PEOPLE = [
    "Rahim Uddin", "Karim Mia", "Fatema Begum", "Shafiqul Islam", "Nasrin Akter",
    "Habibur Rahman", "Tanvir Ahmed", "Mim Sultana", "Jahid Hasan",
    "Sumaiya Islam", "Rakibul Hasan", "Nusrat Jahan", "Arif Hossain",
    "Sadia Afrin", "Mehedi Hasan", "Sabrina Rahman", "Rubel Chowdhury",
]
BKASH_RECHARGE = [
    "Grameenphone Recharge", "Banglalink Recharge", "Robi Recharge",
    "Airtel BD Recharge", "Teletalk Recharge", "GP Star 4G Recharge",
]
BKASH_PAYBILL = [
    "DESCO Bill Payment", "Wasa Dhaka Bill", "DPS Online",
    "Grameenphone Bill Pay", "BTCL Bill", "Titas Gas Bill",
    "DPDC Bill Payment", "Citycell Bill",
]
BKASH_BANKS = [
    "Dutch-Bangla Bank Ltd", "Brac Bank Ltd", "City Bank Ltd",
    "Islami Bank BD", "Pubali Bank Ltd", "Sonali Bank Ltd",
]
BKASH_REMITTANCE = [
    "Western Union Remittance", "Ria Money Transfer",
    "MoneyGram BD", "Xpress Money BD",
]
BKASH_DISBURSEMENT = [
    "DBBL Disbursement", "Brac Microfinance", "Al-Arafah Disbursement",
    "Meridian Housing Disbursement",
]
BKASH_CASHBACK = [
    "bKash Cashback - ShopUp", "MFS Cashback Offer",
    "bKash Rewards", "Cashback - Daraz",
]


def _bkash_fee(ttype, amount):
    if ttype == "Cash Out":
        return max(Decimal("10.00"), q2(amount * Decimal("0.0185")))
    if ttype == "Send Money":
        if amount <= Decimal("2500"):
            return Decimal("5.00")
        return q2(amount * Decimal("0.015"))
    return Decimal("0.00")


def _pick_counterparty(rng, ttype):
    if ttype == "Cash Out":
        return rng.choice(BKASH_CASH_OUT_AGENTS)
    if ttype == "Cash In":
        return rng.choice(BKASH_CASH_IN_AGENTS)
    if ttype == "Send Money":
        return rng.choice(BKASH_SEND_PEOPLE)
    if ttype == "Mobile Recharge":
        return rng.choice(BKASH_RECHARGE)
    if ttype == "Pay Bill":
        return rng.choice(BKASH_PAYBILL)
    if ttype in ("bKash to Bank", "Bank to bKash"):
        return "{} - A/C ****{}".format(rng.choice(BKASH_BANKS), rng.randint(1000, 9999))
    if ttype == "Remittance Received":
        return rng.choice(BKASH_REMITTANCE)
    if ttype == "Disbursement Received":
        return rng.choice(BKASH_DISBURSEMENT)
    if ttype == "Cashback":
        return rng.choice(BKASH_CASHBACK)
    return rng.choice(BKASH_MERCHANTS)  # Make Payment


def _busy_hour(rng):
    return rng.choices(
        [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        weights=[1, 2, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 1, 1],
    )[0]


def generate_bkash(seed, fonts):
    rng = random.Random(subseed(seed, "bkash"))
    account = phone(rng)
    customer = rng.choice([
        "MD. ARIF HOSSAIN", "SABINA YASMIN", "MD. TANVIR AHMED", "NUSRAT JAHAN",
        "MD. RAKIBUL HASAN", "FARHANA AKTER",
    ])
    period_start = "01 July 2026"
    period_end = "31 July 2026"
    issue_date = "05 Aug 2026"
    n = rng.randint(25, 40)

    # Force every type at least once, then fill the remainder by weight.
    types = list(BKASH_TYPES)
    pool = []
    for t, w in BKASH_WEIGHTS:
        pool.extend([t] * w)
    while len(types) < n:
        types.append(rng.choice(pool))
    rng.shuffle(types)

    datetimes = []
    for _ in range(n):
        datetimes.append(datetime(2026, 7, rng.randint(1, 28), _busy_hour(rng), rng.randint(0, 59)))
    datetimes.sort()

    raw = []
    total_out = Decimal("0.00")
    total_in = Decimal("0.00")
    for dt, ttype in zip(datetimes, types):
        lo, hi, step = BKASH_RANGES[ttype]
        amt = Decimal(round(rng.randint(lo, hi) / step) * step)
        fee = _bkash_fee(ttype, amt)
        counterparty = _pick_counterparty(rng, ttype)
        counter_acct = phone(rng)
        tid = trx_id(rng, BKASH_TRX_PREFIX[ttype])
        if ttype in ("Cash In", "Cashback", "Disbursement Received",
                     "Bank to bKash", "Remittance Received"):
            out, inn = Decimal("0.00"), amt
            total_in += amt
        else:
            out, inn = amt, Decimal("0.00")
            total_out += amt
        details = f"{counterparty} / {counter_acct} / TRX ID: {tid}"
        raw.append({
            "date_time": dt.strftime("%d %b %Y, %I:%M %p"),
            "date": dt.strftime("%d %b %Y"),
            "time": dt.strftime("%I:%M %p"),
            "type": ttype,
            "counterparty": counterparty,
            "counterparty_account": counter_acct,
            "trx_id": tid,
            "details_line": details,
            "out": fmt2(out), "out_value": float(out),
            "in": fmt2(inn), "in_value": float(inn),
            "fee": fmt2(fee), "fee_value": float(fee),
            "delta": inn - (out + fee),
        })
    # Two-pass running balance: pick an opening balance large enough that the
    # account never dips below zero (realistic; matches daily bKash usage).
    cum, min_cum = Decimal("0.00"), Decimal("0.00")
    for r in raw:
        cum += r["delta"]
        min_cum = min(min_cum, cum)
    opening = max(Decimal("20000"), -min_cum + Decimal("5000"))
    balance = opening
    for r in raw:
        balance += r["delta"]
        r["balance_after"] = fmt2(balance)
        r["balance_after_value"] = float(balance)
    txns = [{k: v for k, v in r.items() if k != "delta"} for r in raw]

    schema = {
        "_meta": {
            "generator": "src/scripts/generate_bd_samples.py",
            "doc": "bkash_statement",
            "project": "SD-06 PDF Tables (Bangladeshi extension)",
            "seed": seed,
            "font": fonts,
            "format": "PDF (A4, portrait)",
            "ground_truth": "exact - every value below is exactly what is rendered",
        },
        "_notes": {
            "details_format": "{Counterparty} / {01XXXXXXXXX} / TRX ID: {<2-letter type prefix><8 alnum>}",
            "trx_id_prefix_map": BKASH_TRX_PREFIX,
            "fee_formula": {
                "Cash Out": "max(10.00, 1.85% of amount)",
                "Send Money": "flat 5.00 BDT if amount <= 2,500 else 1.5% of amount",
                "others": "0.00",
            },
            "empty_columns_shown_as": "0.00",
            "script": "English, western digits, thousands separators",
            "count_requirement": "25-40 transactions",
        },
        "account": {
            "customer_name": customer,
            "account_number": account,
            "account_number_digits": 11,
            "account_type": "Personal",
            "status": "Active",
            "currency": "BDT",
            "statement_period": f"{period_start} to {period_end}",
            "statement_period_start": period_start,
            "statement_period_end": period_end,
            "issue_date": issue_date,
            "opening_balance": fmt2(opening),
            "opening_balance_value": float(opening),
        },
        "overview": {
            "total_out": fmt2(total_out),
            "total_out_value": float(total_out),
            "total_in": fmt2(total_in),
            "total_in_value": float(total_in),
        },
        "columns": ["Date & Time", "Transaction Type", "Transaction Details",
                    "Out", "In", "Charge/Fee", "Balance"],
        "transactions": txns,
    }

    from reportlab.lib import colors
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable)

    PINK = HexColor("#E2136E")
    LIGHT = HexColor("#FCE7F0")
    S = {
        "title": ParagraphStyle("t", fontName="LATB", fontSize=17, leading=20,
                                alignment=TA_CENTER, textColor=PINK, spaceAfter=1),
        "sub": ParagraphStyle("s", fontName="LAT", fontSize=8.5, leading=11,
                              alignment=TA_CENTER, textColor=colors.HexColor("#5A5A5A")),
        "lbl": ParagraphStyle("l", fontName="LATB", fontSize=8, leading=10.5,
                              textColor=colors.HexColor("#444444")),
        "val": ParagraphStyle("v", fontName="LAT", fontSize=8, leading=10.5),
        "hdr": ParagraphStyle("h", fontName="LATB", fontSize=6.8, leading=8.4,
                              textColor=white),
        "cell": ParagraphStyle("c", fontName="LAT", fontSize=6.8, leading=8.4),
        "cellc": ParagraphStyle("cc", fontName="LAT", fontSize=6.8, leading=8.4,
                                alignment=TA_CENTER),
        "cellr": ParagraphStyle("cr", fontName="LAT", fontSize=6.8, leading=8.4,
                                alignment=TA_RIGHT),
        "ovl": ParagraphStyle("ov", fontName="LATB", fontSize=10, leading=13,
                              alignment=TA_CENTER),
        "ovv": ParagraphStyle("ovv", fontName="LATB", fontSize=12.5, leading=15,
                              alignment=TA_CENTER, textColor=PINK),
    }
    out_path = SD06 / "bkash_statement.pdf"
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=30, rightMargin=30, topMargin=30,
                            bottomMargin=28,
                            title="bKash Statement", author="generate_bd_samples.py")
    story = []
    story.append(Paragraph(markup("bKash Statement"), S["title"]))
    story.append(HRFlowable(width="100%", thickness=1.4, color=PINK, spaceBefore=2, spaceAfter=8))

    info_rows = [
        [Paragraph("Customer Name", S["lbl"]), Paragraph(markup(customer), S["val"]),
         Paragraph("Account No", S["lbl"]), Paragraph(markup(account), S["val"])],
        [Paragraph("Statement Period", S["lbl"]),
         Paragraph(markup(f"{period_start} to {period_end}"), S["val"]),
         Paragraph("Account Type", S["lbl"]), Paragraph("Personal", S["val"])],
        [Paragraph("Issue Date", S["lbl"]), Paragraph(markup(issue_date), S["val"]),
         Paragraph("Status", S["lbl"]), Paragraph("Active", S["val"])],
    ]
    info_tbl = Table(info_rows, colWidths=[85, 175, 85, 175])
    info_tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 8))

    ov = Table(
        [[Paragraph("Total Out (This Period)", S["ovl"]),
          Paragraph("Total In (This Period)", S["ovl"])],
         [Paragraph(markup(f"BDT {fmt2(total_out)}"), S["ovv"]),
          Paragraph(markup(f"BDT {fmt2(total_in)}"), S["ovv"])]],
        colWidths=[265, 265],
    )
    ov.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("BACKGROUND", (1, 0), (1, -1), HexColor("#E3F4E3")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(ov)
    story.append(Spacer(1, 10))

    header = [Paragraph(markup(h), S["hdr"]) for h in
              ["Date & Time", "Transaction Type", "Transaction Details",
               "Out", "In", "Charge/Fee", "Balance"]]
    rows = [header]
    for tx in txns:
        rows.append([
            Paragraph(markup(tx["date_time"]), S["cell"]),
            Paragraph(markup(tx["type"]), S["cell"]),
            Paragraph(markup(tx["details_line"]), S["cell"]),
            Paragraph(tx["out"], S["cellr"]),
            Paragraph(tx["in"], S["cellr"]),
            Paragraph(tx["fee"], S["cellr"]),
            Paragraph(tx["balance_after"], S["cellr"]),
        ])
    tbl = Table(rows, colWidths=[70, 62, 176, 58, 56, 52, 61], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PINK),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F7F7F7")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        markup("This is a computer generated statement and does not require any "
               "signature. For any query, call 16247."), S["sub"]))
    doc.build(story)
    return out_path, schema

# --------------------------------------------------------------------------
# Nagad statement (SD-06)
# --------------------------------------------------------------------------

NAGAD_TYPES = ["CASH IN", "CASH OUT", "SEND MONEY", "MOBILE RECHARGE",
               "PAY BILL", "GOVT DISBURSEMENT"]
NAGAD_RANGES = {
    "CASH IN": (200, 20000, 100),
    "CASH OUT": (200, 15000, 100),
    "SEND MONEY": (100, 2500, 50),
    "MOBILE RECHARGE": (50, 1000, 10),
    "PAY BILL": (300, 10000, 100),
    "GOVT DISBURSEMENT": (1000, 15000, 500),
}
NAGAD_WEIGHTS = [("CASH IN", 3), ("CASH OUT", 4), ("SEND MONEY", 4),
                 ("MOBILE RECHARGE", 2), ("PAY BILL", 2), ("GOVT DISBURSEMENT", 2)]
NAGAD_PEOPLE = [
    "Mohammad Ali", "Rina Begum", "Selim Reza", "Ayesha Siddika", "Faruk Hossain",
    "Khaleda Akter", "Nayeem Islam", "Tania Rahman", "Babul Miah", "Sonia Yeasmin",
]
NAGAD_MERCHANTS = [
    "Shwapno - Dhanmondi", "Agora - Uttara", "Daraz BD", "Foodpanda Bangladesh",
    "Chillox - Dhanmondi", "Sundarban Courier", "Pathao", "Nagad Shop - Motijheel",
    "Meena Bazar - Mirpur", "Bikroy.com",
]
NAGAD_RECHARGE = [
    "Grameenphone Recharge", "Banglalink Recharge", "Robi Recharge",
    "Airtel Recharge", "Teletalk Recharge",
]
NAGAD_PAYBILL = ["DESCO Bill", "DPDC Bill", "Wasa Bill", "Titas Gas Bill", "DPS Bill"]
NAGAD_GOVT = [
    "Cash Transfer (CTF)", "Education Stipend (HSC)", "Freedomb Fighter Allowance",
    "Widow Allowance (VGD)", "Boi Pustok Allowance", "Old Age Allowance",
]
NAGAD_CASH_OUT_AGENTS = [
    "Nagad Agent - Gulshan", "Nagad Agent - Mirpur 10", "Nagad Agent - Uttara Sector 6",
    "Agent Point - Badda", "Cash Out Agent - Banani", "Nagad Agent - Mohammadpur",
]
NAGAD_CASH_IN_AGENTS = [
    "Nagad Cash In - Agent", "Cash In Agent - Dhanmondi",
    "Cash In Point - Motijheel", "Nagad Agent - Khilgaon",
]


def _nagad_counterparty(rng, ttype):
    if ttype == "CASH OUT":
        return rng.choice(NAGAD_CASH_OUT_AGENTS)
    if ttype == "CASH IN":
        return rng.choice(NAGAD_CASH_IN_AGENTS)
    if ttype == "SEND MONEY":
        return rng.choice(NAGAD_PEOPLE)
    if ttype == "MOBILE RECHARGE":
        return rng.choice(NAGAD_RECHARGE)
    if ttype == "PAY BILL":
        return rng.choice(NAGAD_PAYBILL)
    return rng.choice(NAGAD_GOVT)


def generate_nagad(seed, fonts):
    rng = random.Random(subseed(seed, "nagad"))
    account = phone(rng)
    name = rng.choice(["MST. RAHIMA KHATUN", "MD. SAIFUL ISLAM", "AYESHA SIDDIKA",
                       "MD. NAYEEM ISLAM", "SELIM REZA", "RINA BEGUM"])
    n = rng.randint(25, 40)

    types = list(NAGAD_TYPES)
    pool = []
    for t, w in NAGAD_WEIGHTS:
        pool.extend([t] * w)
    while len(types) < n:
        types.append(rng.choice(pool))
    rng.shuffle(types)

    datetimes = []
    for _ in range(n):
        datetimes.append(datetime(2026, 7, rng.randint(1, 28), _busy_hour(rng), rng.randint(0, 59)))
    datetimes.sort()

    raw = []
    total_debit = Decimal("0.00")
    total_credit = Decimal("0.00")
    sl = 0
    for dt, ttype in zip(datetimes, types):
        lo, hi, step = NAGAD_RANGES[ttype]
        amt = Decimal(round(rng.randint(lo, hi) / step) * step)
        counter = _nagad_counterparty(rng, ttype)
        tid = "".join(rng.choices(ALNUM, k=8))
        dstr = dt.strftime("%d/%m/%Y")
        tstr = dt.strftime("%I:%M %p")

        if ttype in ("CASH IN", "GOVT DISBURSEMENT"):
            total_credit += amt
            sl += 1
            raw.append({
                "sl": sl, "txn_date": dstr, "time": tstr, "txn_id": tid,
                "txn_type": ttype, "txn_with": counter, "dr_cr": "CREDIT",
                "amount": fmt2(amt), "amount_value": float(amt),
                "service_fee_row": False, "delta": amt,
            })
        elif ttype == "CASH OUT":
            fee = max(Decimal("10.00"), q2(amt * Decimal("0.0185")))
            total_debit += amt
            sl += 1
            raw.append({
                "sl": sl, "txn_date": dstr, "time": tstr, "txn_id": tid,
                "txn_type": "CASH OUT", "txn_with": counter, "dr_cr": "DEBIT",
                "amount": fmt2(amt), "amount_value": float(amt),
                "service_fee_row": False, "delta": -amt,
            })
            sl += 1
            raw.append({
                "sl": sl, "txn_date": dstr, "time": tstr, "txn_id": tid,
                "txn_type": "CASH OUT(-SERVICE_FEE)", "txn_with": counter,
                "dr_cr": "DEBIT", "amount": fmt2(fee), "amount_value": float(fee),
                "service_fee_row": True, "delta": -fee,
            })
        else:
            total_debit += amt
            sl += 1
            raw.append({
                "sl": sl, "txn_date": dstr, "time": tstr, "txn_id": tid,
                "txn_type": ttype, "txn_with": counter, "dr_cr": "DEBIT",
                "amount": fmt2(amt), "amount_value": float(amt),
                "service_fee_row": False, "delta": -amt,
            })
    # Two-pass running balance (Nagad card balance must never go negative).
    cum, min_cum = Decimal("0.00"), Decimal("0.00")
    for r in raw:
        cum += r["delta"]
        min_cum = min(min_cum, cum)
    opening = max(Decimal("20000"), -min_cum + Decimal("5000"))
    balance = opening
    for r in raw:
        balance += r["delta"]
        r["balance_after"] = fmt2(balance)
        r["balance_after_value"] = float(balance)
    rows = [{k: v for k, v in r.items() if k != "delta"} for r in raw]

    schema = {
        "_meta": {
            "generator": "src/scripts/generate_bd_samples.py",
            "doc": "nagad_statement",
            "project": "SD-06 PDF Tables (Bangladeshi extension)",
            "seed": seed,
            "font": fonts,
            "format": "PDF (A4, portrait)",
            "ground_truth": "exact - every value below is exactly what is rendered",
        },
        "_notes": {
            "cash_out_service_fee": "shown as a separate 'CASH OUT(-SERVICE_FEE)' "
                                    "DEBIT row (fee = max(10.00, 1.85% of amount))",
            "txn_id_format": "8-character alphanumeric (A-Z, 0-9)",
            "dr_cr_values": "DEBIT (money out) / CREDIT (money in)",
            "count_requirement": "25-40 transactions (service-fee rows included)",
        },
        "account": {
            "name": name,
            "account_number": account,
            "account_number_digits": 11,
            "account_type": "CUSTOMER",
            "status": "ACTIVE",
            "kyc_status": "Verified",
            "currency": "BDT",
            "statement_period": "01 July 2026 to 31 July 2026",
            "statement_period_start": "01 July 2026",
            "statement_period_end": "31 July 2026",
            "opening_balance": fmt2(opening),
            "opening_balance_value": float(opening),
        },
        "columns": ["Sl.", "Txn Date", "Time", "Txn ID", "Txn Type", "Txn with",
                    "Dr/Cr", "Amount", "Balance"],
        "summary": {
            "total_debit": fmt2(total_debit),
            "total_debit_value": float(total_debit),
            "total_credit": fmt2(total_credit),
            "total_credit_value": float(total_credit),
            "card_balance": fmt2(balance),
            "card_balance_value": float(balance),
        },
        "transactions": rows,
    }

    from reportlab.lib import colors
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable)

    ORANGE = HexColor("#F26522")
    S = {
        "brand": ParagraphStyle("b", fontName="LATB", fontSize=13, leading=15,
                                alignment=TA_CENTER, textColor=ORANGE, spaceAfter=3),
        "title": ParagraphStyle("t", fontName="LATB", fontSize=18, leading=21,
                                alignment=TA_CENTER, textColor=colors.HexColor("#333333"),
                                spaceAfter=2),
        "lbl": ParagraphStyle("l", fontName="LATB", fontSize=8, leading=10.5,
                              textColor=colors.HexColor("#444444")),
        "val": ParagraphStyle("v", fontName="LAT", fontSize=8, leading=10.5),
        "hdr": ParagraphStyle("h", fontName="LATB", fontSize=6.5, leading=8.2,
                              textColor=white),
        "cell": ParagraphStyle("c", fontName="LAT", fontSize=6.5, leading=8.2),
        "cellc": ParagraphStyle("cc", fontName="LAT", fontSize=6.5, leading=8.2,
                                alignment=TA_CENTER),
        "cellr": ParagraphStyle("cr", fontName="LAT", fontSize=6.5, leading=8.2,
                                alignment=TA_RIGHT),
        "footer": ParagraphStyle("f", fontName="LATB", fontSize=10.5, leading=13,
                                 alignment=TA_RIGHT, textColor=ORANGE),
        "sub": ParagraphStyle("s", fontName="LAT", fontSize=8, leading=10,
                              alignment=TA_CENTER, textColor=colors.HexColor("#5A5A5A")),
    }
    out_path = SD06 / "nagad_statement.pdf"
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=30, rightMargin=30, topMargin=30,
                            bottomMargin=28,
                            title="Nagad Statement of Account",
                            author="generate_bd_samples.py")
    story = []
    story.append(Paragraph(markup("NAGAD"), S["brand"]))
    story.append(Paragraph("Statement of Account", S["title"]))
    story.append(HRFlowable(width="100%", thickness=1.4, color=ORANGE, spaceBefore=2, spaceAfter=8))

    info_rows = [
        [Paragraph("Name", S["lbl"]), Paragraph(markup(name), S["val"]),
         Paragraph("Account No.", S["lbl"]), Paragraph(markup(account), S["val"])],
        [Paragraph("Account Type", S["lbl"]), Paragraph("CUSTOMER", S["val"]),
         Paragraph("Status", S["lbl"]), Paragraph("ACTIVE", S["val"])],
        [Paragraph("KYC Status", S["lbl"]), Paragraph("Verified", S["val"]),
         Paragraph("Statement Period", S["lbl"]),
         Paragraph("01 July 2026 to 31 July 2026", S["val"])],
    ]
    info_tbl = Table(info_rows, colWidths=[85, 175, 85, 175])
    info_tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 10))

    header = [Paragraph(markup(h), S["hdr"]) for h in
              ["Sl.", "Txn Date", "Time", "Txn ID", "Txn Type", "Txn with",
               "Dr/Cr", "Amount", "Balance"]]
    trows = [header]
    for r in rows:
        trows.append([
            Paragraph(str(r["sl"]), S["cellc"]),
            Paragraph(r["txn_date"], S["cellc"]),
            Paragraph(r["time"], S["cellc"]),
            Paragraph(r["txn_id"], S["cellc"]),
            Paragraph(r["txn_type"], S["cell"]),
            Paragraph(markup(r["txn_with"]), S["cell"]),
            Paragraph(r["dr_cr"], S["cellc"]),
            Paragraph(r["amount"], S["cellr"]),
            Paragraph(r["balance_after"], S["cellr"]),
        ])
    tbl = Table(trows, colWidths=[22, 52, 42, 66, 94, 96, 38, 60, 65], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ORANGE),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#FFF4EE")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))
    story.append(Paragraph(markup(f"Card Balance: BDT {fmt2(balance)}"), S["footer"]))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        markup("This is an electronically generated statement. For any query "
               "contact the Nagad helpline (16216)."), S["sub"]))
    doc.build(story)
    return out_path, schema

# --------------------------------------------------------------------------
# Mushak 6.3 VAT invoice (SD-08)
# --------------------------------------------------------------------------

def generate_mushak(seed, fonts):
    rng = random.Random(subseed(seed, "mushak"))
    items = [
        {"desc": "Canned Soft Drinks (250 ml x 12)", "unit": "Carton", "qty": 50,
         "unit_value": Decimal("480.00"), "sd_rate": Decimal("0.20"),
         "vat_rate": Decimal("0.15"), "specific_tax": Decimal("0.00")},
        {"desc": "Assorted Biscuits", "unit": "Carton", "qty": 120,
         "unit_value": Decimal("320.00"), "sd_rate": Decimal("0.00"),
         "vat_rate": Decimal("0.15"), "specific_tax": Decimal("0.00")},
        {"desc": "Laundry Soap 100 g", "unit": "Box", "qty": 300,
         "unit_value": Decimal("45.00"), "sd_rate": Decimal("0.00"),
         "vat_rate": Decimal("0.15"), "specific_tax": Decimal("0.00")},
        {"desc": "Cotton Textile Fabric", "unit": "Meter", "qty": 500,
         "unit_value": Decimal("120.00"), "sd_rate": Decimal("0.00"),
         "vat_rate": Decimal("0.15"), "specific_tax": Decimal("0.00")},
        {"desc": "Footwear (Sandals)", "unit": "Pair", "qty": 80,
         "unit_value": Decimal("650.00"), "sd_rate": Decimal("0.10"),
         "vat_rate": Decimal("0.15"), "specific_tax": Decimal("0.00")},
        {"desc": "Restaurant Section - Food & Beverage Service", "unit": "Set",
         "qty": 40, "unit_value": Decimal("250.00"), "sd_rate": Decimal("0.00"),
         "vat_rate": Decimal("0.05"), "specific_tax": Decimal("0.00")},
    ]
    lines = []
    total_value = Decimal("0.00")
    total_sd = Decimal("0.00")
    total_vat = Decimal("0.00")
    total_specific = Decimal("0.00")
    for i, it in enumerate(items, start=1):
        tv = q2(it["qty"] * it["unit_value"])
        sd = q2(tv * it["sd_rate"])
        vat = q2(tv * it["vat_rate"])
        stax = q2(it["specific_tax"])
        total_value += tv
        total_sd += sd
        total_vat += vat
        total_specific += stax
        lines.append({
            "sl": i,
            "description": it["desc"],
            "unit_of_qty": f"{it['qty']} {it['unit']}",
            "qty": it["qty"],
            "unit": it["unit"],
            "unit_value": fmt2(it["unit_value"]),
            "unit_value_value": float(it["unit_value"]),
            "total_value": fmt2(tv),
            "total_value_value": float(tv),
            "supplementary_duty": fmt2(sd),
            "supplementary_duty_value": float(sd),
            "vat_rate": f"{int(it['vat_rate'] * 100)}%",
            "vat_rate_value": float(it["vat_rate"]),
            "specific_tax": fmt2(stax),
            "specific_tax_value": float(stax),
            "vat_amount": fmt2(vat),
            "vat_amount_value": float(vat),
        })
    grand = total_value + total_sd + total_vat + total_specific

    invoice = {
        "invoice_no": "M63-2026-0417",
        "date": "12 Jul 2026",
        "time": "10:45 AM",
        "destination_of_supply": "Chattogram",
    }
    registered = {
        "name": "Dhaka Trading & Distribution Limited",
        "bin": "001234567",
        "address": "75 Motijheel C/A, Dhaka 1000",
        "phone": "+880-2-9551234",
    }
    purchaser = {
        "name": "Agrabad Super Mart",
        "bin": "001998877",
        "address": "1027 Agrabad C/A, Chattogram 4000",
    }
    officer = {
        "designation": "VAT Officer",
        "name": "Md. Sirajul Islam",
        "station": "VAT Circle - 11, Dhaka",
    }

    schema = {
        "_meta": {
            "generator": "src/scripts/generate_bd_samples.py",
            "doc": "mushak63_invoice",
            "project": "SD-08 Invoices (Bangladeshi extension)",
            "seed": seed,
            "font": fonts,
            "format": "PDF (A4, portrait)",
            "ground_truth": "exact - every value below is exactly what is rendered",
        },
        "_notes": {
            "form": "MUSHAK 6.3 Tax Challan [Rule 40] (bilingual header, Bengali + English)",
            "vat_rates": "15% standard (goods), 5% restaurant section line",
            "supplementary_duty": "20% on soft drinks, 10% on footwear, 0% otherwise",
            "specific_tax": "0.00 on all lines (column present per plan schema)",
            "grand_total_formula": "sum(Total Value) + sum(Supplementary Duty) + "
                                   "sum(VAT amount) + sum(Specific Tax)",
        },
        "registered_person": registered,
        "purchaser": purchaser,
        "invoice": invoice,
        "columns": ["SL", "Goods/Service Description", "Unit of Qty", "Unit Value",
                    "Total Value", "Supplementary Duty", "VAT rate", "Specific Tax",
                    "VAT amount"],
        "line_items": lines,
        "totals": {
            "total_value": fmt2(total_value),
            "total_value_value": float(total_value),
            "total_supplementary_duty": fmt2(total_sd),
            "total_supplementary_duty_value": float(total_sd),
            "total_vat": fmt2(total_vat),
            "total_vat_value": float(total_vat),
            "total_specific_tax": fmt2(total_specific),
            "total_specific_tax_value": float(total_specific),
            "grand_total": fmt2(grand),
            "grand_total_value": float(grand),
        },
        "officer_in_charge": officer,
    }

    from reportlab.lib import colors
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable)

    GREEN = HexColor("#1A7A30")
    LIGHT_G = HexColor("#EAF4EC")
    S = {
        "govt": ParagraphStyle("g", fontName="LAT", fontSize=8.5, leading=11,
                               alignment=TA_CENTER, textColor=colors.HexColor("#444444")),
        "bn_title": ParagraphStyle("bt", fontName="BNB", fontSize=15, leading=19,
                                   alignment=TA_CENTER, textColor=GREEN, spaceAfter=1),
        "en_title": ParagraphStyle("et", fontName="LATB", fontSize=13, leading=16,
                                   alignment=TA_CENTER, textColor=GREEN, spaceAfter=6),
        "box_title": ParagraphStyle("bx", fontName="LATB", fontSize=8.5, leading=11,
                                    textColor=colors.HexColor("#333333")),
        "lbl": ParagraphStyle("l", fontName="LATB", fontSize=8, leading=10.5,
                              textColor=colors.HexColor("#444444")),
        "val": ParagraphStyle("v", fontName="LAT", fontSize=8, leading=10.5),
        "hdr": ParagraphStyle("h", fontName="LATB", fontSize=6.2, leading=7.8,
                              textColor=white),
        "cell": ParagraphStyle("c", fontName="LAT", fontSize=6.6, leading=8.2),
        "cellc": ParagraphStyle("cc", fontName="LAT", fontSize=6.6, leading=8.2,
                                alignment=TA_CENTER),
        "cellr": ParagraphStyle("cr", fontName="LAT", fontSize=6.6, leading=8.2,
                                alignment=TA_RIGHT),
        "totl": ParagraphStyle("tl", fontName="LAT", fontSize=8, leading=10,
                               alignment=TA_RIGHT),
        "totb": ParagraphStyle("tb", fontName="LATB", fontSize=9, leading=11,
                               alignment=TA_RIGHT, textColor=GREEN),
        "sigs": ParagraphStyle("ss", fontName="LATB", fontSize=8.5, leading=11,
                               alignment=TA_CENTER),
        "sig": ParagraphStyle("sg", fontName="LAT", fontSize=8.5, leading=11,
                              alignment=TA_CENTER),
    }
    out_path = SD08 / "mushak63_invoice.pdf"
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=30, rightMargin=30, topMargin=30,
                            bottomMargin=28,
                            title="MUSHAK 6.3 Tax Challan",
                            author="generate_bd_samples.py")
    story = []
    story.append(Paragraph(markup("Government of the People's Republic of Bangladesh"),
                           S["govt"]))
    story.append(Paragraph(markup("জাতীয় রাজস্ব বোর্ড", bn="BNB"), S["govt"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(markup("মূসক ৬.৩ কর দাখিলা [বিধি ৪০]", bn="BNB"), S["bn_title"]))
    story.append(Paragraph(markup("MUSHAK 6.3 TAX CHALLAN [RULE 40]"), S["en_title"]))
    story.append(HRFlowable(width="100%", thickness=1.4, color=GREEN, spaceBefore=2, spaceAfter=8))

    def box(title, rows):
        data = [[Paragraph(markup(title), S["box_title"])]] + rows
        t = Table(data, colWidths=[160, 150])
        t.setStyle(TableStyle([
            ("SPAN", (0, 0), (-1, 0)),
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT_G),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        return t

    reg_rows = [
        [Paragraph("Registered Person's Name", S["lbl"]),
         Paragraph(markup(registered["name"]), S["val"])],
        [Paragraph("BIN (Business Identification Number)", S["lbl"]),
         Paragraph(markup(registered["bin"]), S["val"])],
        [Paragraph("Address", S["lbl"]),
         Paragraph(markup(registered["address"]), S["val"])],
    ]
    pur_rows = [
        [Paragraph("Purchaser's Name", S["lbl"]),
         Paragraph(markup(purchaser["name"]), S["val"])],
        [Paragraph("BIN", S["lbl"]),
         Paragraph(markup(purchaser["bin"]), S["val"])],
        [Paragraph("Address", S["lbl"]),
         Paragraph(markup(purchaser["address"]), S["val"])],
    ]
    party = Table(
        [[box("Registered Person (বিক্রেতা)", reg_rows),
          box("Purchaser (ক্রেতা)", pur_rows)]],
        colWidths=[315, 315])
    party.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(party)
    story.append(Spacer(1, 8))

    inv_rows = [
        [Paragraph("Invoice No", S["lbl"]), Paragraph(markup(invoice["invoice_no"]), S["val"]),
         Paragraph("Date", S["lbl"]), Paragraph(markup(invoice["date"]), S["val"])],
        [Paragraph("Time", S["lbl"]), Paragraph(markup(invoice["time"]), S["val"]),
         Paragraph("Destination of Supply", S["lbl"]),
         Paragraph(markup(invoice["destination_of_supply"]), S["val"])],
    ]
    inv_tbl = Table(inv_rows, colWidths=[70, 135, 90, 135])
    inv_tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(inv_tbl)
    story.append(Spacer(1, 10))

    header = [Paragraph(markup(h), S["hdr"]) for h in [
        "SL", "Goods/Service Description", "Unit of Qty", "Unit Value",
        "Total Value", "Supplementary Duty", "VAT rate", "Specific Tax",
        "VAT amount",
    ]]
    trows = [header]
    for ln in lines:
        trows.append([
            Paragraph(str(ln["sl"]), S["cellc"]),
            Paragraph(markup(ln["description"]), S["cell"]),
            Paragraph(ln["unit_of_qty"], S["cellc"]),
            Paragraph(ln["unit_value"], S["cellr"]),
            Paragraph(ln["total_value"], S["cellr"]),
            Paragraph(ln["supplementary_duty"], S["cellr"]),
            Paragraph(ln["vat_rate"], S["cellc"]),
            Paragraph(ln["specific_tax"], S["cellr"]),
            Paragraph(ln["vat_amount"], S["cellr"]),
        ])
    tbl = Table(trows, colWidths=[20, 159, 48, 52, 57, 55, 42, 46, 56], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GREEN),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F2F8F3")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))

    totals = [
        [Paragraph("Total Goods/Service Value", S["totl"]),
         Paragraph(markup(f"BDT {fmt2(total_value)}"), S["totl"])],
        [Paragraph("Total Supplementary Duty", S["totl"]),
         Paragraph(markup(f"BDT {fmt2(total_sd)}"), S["totl"])],
        [Paragraph("Total VAT (15% + 5%)", S["totl"]),
         Paragraph(markup(f"BDT {fmt2(total_vat)}"), S["totl"])],
        [Paragraph("Total Specific Tax", S["totl"]),
         Paragraph(markup(f"BDT {fmt2(total_specific)}"), S["totl"])],
        [Paragraph("GRAND TOTAL", S["totb"]),
         Paragraph(markup(f"BDT {fmt2(grand)}"), S["totb"])],
    ]
    tt = Table(totals, colWidths=[240, 120])
    tt.setStyle(TableStyle([
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEABOVE", (0, 4), (-1, 4), 1, GREEN),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(tt)
    story.append(Spacer(1, 14))

    sig = Table(
        [[Paragraph("Officer-in-Charge", S["sigs"]),
          Paragraph(markup("Received By (প্রাপক)"), S["sigs"])],
         [Paragraph(markup("(বহিস্থ আধিকারিক)", bn="BNB"), S["sigs"]),
          Paragraph("", S["sig"])],
         [Paragraph("", S["sig"]), Paragraph("", S["sig"])],
         [Paragraph(markup(f"Signature & Seal: {officer['name']}"), S["sig"]),
          Paragraph(markup("Signature & Seal"), S["sig"])],
         [Paragraph(markup(f"{officer['designation']}, {officer['station']}"), S["sig"]),
          Paragraph("", S["sig"])]],
        colWidths=[315, 315],
    )
    sig.setStyle(TableStyle([
        ("LINEABOVE", (0, 2), (-1, 2), 0.6, colors.HexColor("#555555")),
        ("LINEABOVE", (0, 1), (-1, 1), 0.3, colors.HexColor("#999999")),
    ]))
    story.append(sig)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        markup("This challan is generated under the Value Added Tax and Supplementary "
               "Duty Act, 2012. ভ্যাট ও সম্পূরক শুল্ক আইন, ২০১২ অনুযায়ী প্রস্তুতকৃত।"),
        S["govt"]))
    doc.build(story)
    return out_path, schema

# --------------------------------------------------------------------------
# Bilingual restaurant bill (SD-08)
# --------------------------------------------------------------------------

MENU = [
    ("Chicken Biryani", "মুরগির বিরিয়ানি", "350"),
    ("Beef Kacchi", "গরুর কাচ্চি", "520"),
    ("Mutton Rezala", "খাসির রেজালা", "460"),
    ("Chicken Roast", "মুরগির রোস্ট", "280"),
    ("Shahi Pulao", "শাহী পোলাও", "240"),
    ("Kacchi Biryani", "কাচ্চি বিরিয়ানি", "420"),
    ("Borhani", "বোরহানি", "80"),
    ("Misti Doi", "মিষ্টি দই", "120"),
    ("Bhorta Set", "ভর্তা সেট", "180"),
    ("Rui Fish Curry", "রুই মাছের ঝোল", "320"),
    ("Chicken Fried Rice", "চিকেন ফ্রাইড রাইস", "220"),
    ("Chicken Chowmein", "চিকেন চাউমিন", "190"),
    ("Mineral Water", "মিনারেল ওয়াটার", "30"),
    ("Lassi", "লাচ্ছি", "90"),
    ("Beef Kabab", "গরুর কাবাব", "150"),
]


def generate_restaurant(seed, fonts):
    rng = random.Random(subseed(seed, "restaurant"))
    n_items = rng.randint(5, 8)
    chosen = rng.sample(MENU, n_items)
    items = []
    subtotal = Decimal("0.00")
    for sl, (en, bn, rate) in enumerate(chosen, start=1):
        qty = rng.choices([1, 1, 2, 2, 3], weights=[3, 2, 2, 1, 1])[0]
        rate_d = Decimal(rate)
        amt = q2(qty * rate_d)
        subtotal += amt
        items.append({
            "sl": sl,
            "item_en": en,
            "item_bn": bn,
            "item_name": f"{en} ({bn})",
            "qty": qty,
            "unit_price": fmt2(rate_d),
            "unit_price_value": float(rate_d),
            "amount": fmt2(amt),
            "amount_value": float(amt),
        })
    vat = q2(subtotal * Decimal("0.05"))
    service = q2(subtotal * Decimal("0.10"))
    grand = subtotal + vat + service

    restaurant = {
        "name_en": "Cafe Ghorowa",
        "name_bn": "ক্যাফে ঘরোয়া",
        "address_en": "House 42, Road 11, Dhanmondi, Dhaka 1209",
        "address_bn": "বাড়ি ৪২, রোড ১১, ধানমন্ডি, ঢাকা ১২০৯",
        "phone": "+880-2-9674321",
        "bin": "001125678",
    }
    bill = {
        "bill_no": "R-0725-0118",
        "date": "15 Jul 2026",
        "time": "8:45 PM",
        "table": "12",
        "waiter": "Sumon",
        "payment_method": "bKash",
    }
    schema = {
        "_meta": {
            "generator": "src/scripts/generate_bd_samples.py",
            "doc": "bilingual_restaurant_bill",
            "project": "SD-08 Invoices (Bangladeshi extension)",
            "seed": seed,
            "font": fonts,
            "format": "PDF (A4, portrait)",
            "ground_truth": "exact - every value below is exactly what is rendered",
        },
        "_notes": {
            "bilingual": "item names shown as 'English (বাংলা)'",
            "vat_rate": "5%",
            "service_charge_rate": "10%",
            "currency": "BDT (৳)",
            "grand_total_formula": "subtotal + VAT(5%) + service charge(10%)",
        },
        "restaurant": restaurant,
        "bill": bill,
        "columns": ["#", "Item", "Qty", "Unit Price (৳)", "Amount (৳)"],
        "items": items,
        "totals": {
            "subtotal": fmt2(subtotal),
            "subtotal_value": float(subtotal),
            "vat": fmt2(vat),
            "vat_value": float(vat),
            "service_charge": fmt2(service),
            "service_charge_value": float(service),
            "grand_total": fmt2(grand),
            "grand_total_value": float(grand),
        },
    }

    from reportlab.lib import colors
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable)

    BROWN = HexColor("#8B4513")
    S = {
        "title": ParagraphStyle("t", fontName="LATB", fontSize=19, leading=22,
                                alignment=TA_CENTER, textColor=BROWN, spaceAfter=1),
        "title_bn": ParagraphStyle("tb", fontName="BNB", fontSize=13, leading=16,
                                   alignment=TA_CENTER, textColor=BROWN, spaceAfter=2),
        "addr": ParagraphStyle("a", fontName="LAT", fontSize=8.5, leading=11,
                               alignment=TA_CENTER, textColor=colors.HexColor("#555555")),
        "lbl": ParagraphStyle("l", fontName="LATB", fontSize=8.5, leading=11),
        "val": ParagraphStyle("v", fontName="LAT", fontSize=8.5, leading=11),
        "hdr": ParagraphStyle("h", fontName="LATB", fontSize=8, leading=10,
                              textColor=white),
        "cell": ParagraphStyle("c", fontName="LAT", fontSize=8, leading=10),
        "cellc": ParagraphStyle("cc", fontName="LAT", fontSize=8, leading=10,
                                alignment=TA_CENTER),
        "cellr": ParagraphStyle("cr", fontName="LAT", fontSize=8, leading=10,
                                alignment=TA_RIGHT),
        "totl": ParagraphStyle("tl", fontName="LAT", fontSize=9, leading=12,
                               alignment=TA_RIGHT),
        "totb": ParagraphStyle("tb2", fontName="LATB", fontSize=11, leading=13,
                               alignment=TA_RIGHT, textColor=BROWN),
        "foot": ParagraphStyle("f", fontName="LATB", fontSize=10, leading=13,
                               alignment=TA_CENTER, textColor=BROWN),
    }
    out_path = SD08 / "bilingual_restaurant_bill.pdf"
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=44, rightMargin=44, topMargin=36,
                            bottomMargin=32,
                            title="Cafe Ghorowa - Restaurant Bill",
                            author="generate_bd_samples.py")
    story = []
    story.append(Paragraph(markup(restaurant["name_en"]), S["title"]))
    story.append(Paragraph(markup(restaurant["name_bn"], bn="BNB"), S["title_bn"]))
    story.append(Paragraph(markup(restaurant["address_en"]), S["addr"]))
    story.append(Paragraph(markup(restaurant["address_bn"], bn="BNB"), S["addr"]))
    story.append(Paragraph(
        markup(f"Phone: {restaurant['phone']}  |  BIN: {restaurant['bin']}"), S["addr"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=BROWN, spaceBefore=4, spaceAfter=8))

    meta_rows = [
        [Paragraph("Bill No", S["lbl"]), Paragraph(markup(bill["bill_no"]), S["val"]),
         Paragraph("Date", S["lbl"]), Paragraph(markup(bill["date"]), S["val"])],
        [Paragraph("Time", S["lbl"]), Paragraph(markup(bill["time"]), S["val"]),
         Paragraph("Table", S["lbl"]), Paragraph(markup(bill["table"]), S["val"])],
        [Paragraph("Waiter", S["lbl"]), Paragraph(markup(bill["waiter"]), S["val"]),
         Paragraph("Payment", S["lbl"]), Paragraph(markup(bill["payment_method"]), S["val"])],
    ]
    mt = Table(meta_rows, colWidths=[60, 175, 60, 175])
    mt.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(mt)
    story.append(Spacer(1, 10))

    header = [Paragraph(markup(h), S["hdr"]) for h in
              ["#", "Item", "Qty", "Unit Price (৳)", "Amount (৳)"]]
    trows = [header]
    for it in items:
        trows.append([
            Paragraph(str(it["sl"]), S["cellc"]),
            Paragraph(markup(it["item_name"]), S["cell"]),
            Paragraph(str(it["qty"]), S["cellc"]),
            Paragraph(markup(f"৳{it['unit_price']}"), S["cellr"]),
            Paragraph(markup(f"৳{it['amount']}"), S["cellr"]),
        ])
    tbl = Table(trows, colWidths=[22, 265, 40, 80, 100], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BROWN),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#FDF6EE")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))

    totals = [
        [Paragraph("Subtotal", S["totl"]), Paragraph(markup(f"৳{fmt2(subtotal)}"), S["totl"])],
        [Paragraph("VAT (5%)", S["totl"]), Paragraph(markup(f"৳{fmt2(vat)}"), S["totl"])],
        [Paragraph("Service Charge (10%)", S["totl"]),
         Paragraph(markup(f"৳{fmt2(service)}"), S["totl"])],
        [Paragraph("GRAND TOTAL", S["totb"]), Paragraph(markup(f"৳{fmt2(grand)}"), S["totb"])],
    ]
    tt = Table(totals, colWidths=[250, 145])
    tt.setStyle(TableStyle([
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEABOVE", (0, 3), (-1, 3), 1.2, BROWN),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(tt)
    story.append(Spacer(1, 14))
    story.append(Paragraph(markup("Thank you for visiting!  ধন্যবাদ!", bn="BNB"), S["foot"]))
    story.append(Paragraph(
        markup("VAT & Service Charge shown separately. Please pay at the counter."),
        S["addr"]))
    doc.build(story)
    return out_path, schema


# --------------------------------------------------------------------------
# Bengali WhatsApp chat (SD-07)
# --------------------------------------------------------------------------

CHAT_SENDERS = ["Arif", "Sumaiya", "Rahim", "Nusrat"]
CHAT_BANGLA = [
    "ভালো আছি, তুমি কেমন?",
    "আজ রাতে ক্যাম্পাসে দেখা হবে?",
    "লাইব্রেরিতে বসে পড়াশোনা করছি।",
    "কাল সকালে ক্লাস আছে না?",
    "চল সবাই মিলে বাইরে খেয়ে আসি।",
    "মিরপুর ১০-এ একটা নতুন রেস্টুরেন্ট খুলেছে।",
    "পরীক্ষার রুটিন পেয়েছিস?",
    "টাকাটা বিকাশে পাঠিয়ে দিলাম, চেক কর।",
    "আলহামদুলিল্লাহ, রেজাল্ট ভালো হয়েছে!",
    "তুই তো শুধু ফোন নিয়ে ব্যস্ত থাকিস!",
    "আমার কাছে বইটা আছে, কাল আনবো।",
    "বৃষ্টি হচ্ছে, ছাতা নিয়ে আসিস।",
    "গতকালের লেকচার শিটগুলো কে পেয়েছে?",
    "নতুন সিনেমা দেখতে যাব?",
    "দুপুরে ডিম ভাত খেয়েছি, রাতে বিরিয়ানি খাব।",
    "মোবাইল রিচার্জ করলাম, ২০০ টাকা।",
    "সন্ধ্যায় দেখা করব, ঠিক আছে?",
    "আম্মুকে বলেছি, কাল আসবো।",
]
CHAT_BANGLISH = [
    "Kemon achis? Long time no see!",
    "Bhai, eto late keno?",
    "Screenshot ta patha dey naki?",
    "Haha nice one. Bujhte parlam na first e.",
    "Chill, amra shob manage korbo.",
    "Ok sure, I will be there by 6.",
    "Bola to hoy nai... ki hoise?",
    "Actually ami busy chilam. Kal kotha bolbo.",
    "Take care brother, ghumay porte jachchi.",
    "Ei output ta perfect lagche. Great job!",
    "Boro bhai keo nijer moto kore dekhchhe.",
    "Tui kobe return korbi?",
    "Taka send kore dichi, check koro.",
    "Ami basha te chole gechi, baki kal.",
    "Let's fix the meetup for Saturday.",
    "Ei screenshot-e full marks!",
    "Nusrat, tuu kothay? Miss korechi.",
]
CHAT_MEDIA = ["<Media omitted>", "Image omitted", "Sticker", "Location shared",
              "Video omitted", "GIF"]
CHAT_SYSTEM = [
    "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.",
    "Arif added Nusrat",
    "You changed this group's icon",
    "You deleted this message",
    "This message was deleted",
    "Voice call ended (0:42)",
    "Video call ended (3:15)",
    "You removed Rahim",
    "Nusrat left",
]
CHAT_MULTILINE = (
    "বাসা থেকে বের হবো একটু পরেই।\n"
    "মিরপুর থেকে গুলশানে যেতে কতক্ষণ লাগবে?\n"
    "রিকশা নিলে ৪০-৫০ মিনিট, উবারে ৩০ মিনিট।"
)


def generate_chat(seed, fonts):
    rng = random.Random(subseed(seed, "chat"))
    n = rng.randint(30, 50)

    def ts(day, hour, minute):
        return datetime(2026, 7, day, hour, minute)

    slots = []
    for _ in range(n):
        slots.append(ts(rng.randint(14, 17), _busy_hour(rng), rng.randint(0, 59)))

    # Fixed opening system lines for realism.
    records = [
        {"ts": ts(14, 20, 0), "kind": "system",
         "text": "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them."},
        {"ts": ts(14, 20, 1), "kind": "system", "text": "Arif added Nusrat"},
        {"ts": ts(14, 20, 2), "kind": "message", "sender": "Arif",
         "text": "Welcome Nusrat! এখানে সবাই আছে, চিন্তা কইরো না।"},
    ]

    # A conversation body from the pools.
    for slot in slots:
        kind = rng.choices(["message", "message", "message", "media", "system"],
                           weights=[4, 4, 4, 1, 1])[0]
        if kind == "message":
            sender = rng.choice(CHAT_SENDERS)
            pool = CHAT_BANGLA if rng.random() < 0.55 else CHAT_BANGLISH
            records.append({"ts": slot, "kind": "message", "sender": sender,
                            "text": rng.choice(pool)})
        elif kind == "media":
            sender = rng.choice(CHAT_SENDERS + ["You"])
            records.append({"ts": slot, "kind": "media", "sender": sender,
                            "text": rng.choice(CHAT_MEDIA)})
        else:
            records.append({"ts": slot, "kind": "system",
                            "text": rng.choice(CHAT_SYSTEM)})

    # Force the multi-line chunking stressor.
    records.append({"ts": ts(15, 9, 12), "kind": "message", "sender": "Arif",
                    "text": CHAT_MULTILINE, "multiline": True})

    records.sort(key=lambda r: (r["ts"],))  # stable for identical timestamps

    def wts(dt):
        return dt.strftime("%d/%m/%y, %I:%M %p")

    lines_out = []
    messages = []
    for idx, r in enumerate(records, start=1):
        stamp = wts(r["ts"])
        if r["kind"] == "system":
            raw = f"{stamp} - {r['text']}"
            lines_out.append(raw)
            messages.append({
                "no": idx, "raw": raw, "timestamp": stamp,
                "date": r["ts"].strftime("%d/%m/%y"),
                "time": r["ts"].strftime("%I:%M %p"),
                "sender": None, "text": r["text"], "kind": "system",
                "multiline": False,
            })
        else:
            sender = r["sender"]
            msg_kind = "media" if r["kind"] == "media" else "message"
            if r.get("multiline"):
                first, *rest = r["text"].split("\n")
                raw = f"{stamp} - {sender}: {first}"
                lines_out.append(raw)
                lines_out.extend(rest)
                messages.append({
                    "no": idx, "raw": raw + "\n" + "\n".join(rest),
                    "timestamp": stamp, "date": r["ts"].strftime("%d/%m/%y"),
                    "time": r["ts"].strftime("%I:%M %p"),
                    "sender": sender, "text": r["text"], "kind": msg_kind,
                    "multiline": True,
                })
            else:
                raw = f"{stamp} - {sender}: {r['text']}"
                lines_out.append(raw)
                messages.append({
                    "no": idx, "raw": raw, "timestamp": stamp,
                    "date": r["ts"].strftime("%d/%m/%y"),
                    "time": r["ts"].strftime("%I:%M %p"),
                    "sender": sender, "text": r["text"], "kind": msg_kind,
                    "multiline": False,
                })

    schema = {
        "_meta": {
            "generator": "src/scripts/generate_bd_samples.py",
            "doc": "bangla_chat",
            "project": "SD-07 Chat Exports (Bangladeshi extension)",
            "seed": seed,
            "font": fonts,
            "format": "TXT (WhatsApp chat export)",
            "ground_truth": "exact - every line below is exactly what is written",
        },
        "_notes": {
            "format": "DD/MM/YY, h:mm AM/PM - Name: message (real WhatsApp export format)",
            "regex": r"^(\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2} [AP]M) - ([^:]+): (.*)$",
            "system_lines": "timestamp prefix but no 'Name:' part",
            "media_lines": "Name: <Media omitted> / Image omitted / Sticker / Location shared / Video omitted / GIF",
            "multiline_messages": "continuation lines have no timestamp prefix (chunking stressor)",
            "count_requirement": "30-50 messages",
        },
        "participants": CHAT_SENDERS + ["You"],
        "group_name": "Dhaka University Friends",
        "message_count": len(messages),
        "messages": messages,
    }

    out_path = SD07 / "bangla_chat.txt"
    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return out_path, schema

# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

CHAT_MSG_RE = re.compile(
    r"^(\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2} (?:AM|PM)) - ([^:]+): (.*)$"
)

# reportlab stamps a wall-clock CreationDate/ModDate; the D: string is
# fixed-length, so replacing it in place keeps the PDF xref table valid and
# makes the whole output byte-deterministic for a given seed.
_PDF_DATE_RE = re.compile(rb"\(D:\d{14}[+-]\d{2}'\d{2}'\)")
_FIXED_PDF_DATE = b"(D:20260803000000+06'00')"
_PDF_ID_RE = re.compile(rb"(/ID\s*\[\s*)<[0-9a-f]{32}>(\s*)<[0-9a-f]{32}>(\s*\])")
_FIXED_PDF_ID = b"<00000000000000000000000000000000>"


def normalize_pdf_metadata(path):
    """Strike reportlab's wall-clock dates + file ID so PDF bytes are seed-deterministic."""
    data = path.read_bytes()
    data = _PDF_DATE_RE.sub(_FIXED_PDF_DATE, data)
    data = _PDF_ID_RE.sub(lambda m: m.group(1) + _FIXED_PDF_ID + m.group(2)
                          + _FIXED_PDF_ID + m.group(3), data)
    path.write_bytes(data)


def norm(text):
    """Strip ALL whitespace so substring checks survive table-cell wrapping."""
    return re.sub(r"\s+", "", text or "")


def extract_pdf_text(path):
    try:
        import pdfplumber
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pdfplumber"],
                       check=True)
        import pdfplumber
    chunks = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            chunks.append(txt)
    return "\n".join(chunks)


def has_bengali(text):
    return any(0x0980 <= ord(ch) <= 0x09FF for ch in text)


def verify_all(artifacts):
    """artifacts: list of (name, kind, path, schema). Prints PASS/FAIL report."""
    results = []
    ok = True

    for name, kind, path, schema in artifacts:
        if kind == "pdf":
            text = extract_pdf_text(path)
            nt = norm(text)
            results.append((f"{name}: extract_text() non-empty", bool(text.strip())))
            if schema["_meta"]["doc"] in ("mushak63_invoice", "bilingual_restaurant_bill"):
                results.append((f"{name}: contains Bengali chars", has_bengali(text)))
            if schema["_meta"]["doc"] == "bkash_statement":
                txns = schema["transactions"]
                ids_in_text = all(norm(tx["trx_id"]) in nt for tx in txns)
                ci_ids = [tx["trx_id"] for tx in txns if tx["type"] == "Cash In"]
                results.append((f"{name}: header 'bKash Statement'", "bKash Statement" in text))
                results.append((f"{name}: type 'Cash Out' present", "Cash Out" in text))
                results.append((f"{name}: all {len(txns)} TRX IDs extractable", ids_in_text))
                results.append((f"{name}: Cash In 'CI<8 alnum>' TRX IDs present", bool(ci_ids)))
                results.append((f"{name}: txn count 25-40", 25 <= len(txns) <= 40))
            elif schema["_meta"]["doc"] == "nagad_statement":
                rows = schema["transactions"]
                fee_rows = [r for r in rows if r["txn_type"] == "CASH OUT(-SERVICE_FEE)"]
                ids_in_text = all(norm(r["txn_id"]) in nt for r in rows)
                results.append((f"{name}: header 'Statement of Account'",
                                "Statement of Account" in text))
                results.append((f"{name}: 'Card Balance' footer", "Card Balance" in text))
                results.append((f"{name}: 'CASH OUT(-SERVICE_FEE)' rows present",
                                bool(fee_rows) and "OUT(-SERVICE_FEE)" in nt))
                results.append((f"{name}: all {len(rows)} Txn IDs extractable", ids_in_text))
                results.append((f"{name}: txn rows 25-40", 25 <= len(rows) <= 40))
            elif schema["_meta"]["doc"] == "mushak63_invoice":
                results.append((f"{name}: 'MUSHAK 6.3 TAX CHALLAN' present",
                                "MUSHAK 6.3 TAX CHALLAN" in text))
                results.append((f"{name}: 'RULE 40' present", "RULE 40" in text))
                results.append((f"{name}: Bengali 'মূসক' present", "মূসক" in text))
                results.append((f"{name}: 'GRAND TOTAL' present", "GRAND TOTAL" in text))
                results.append((f"{name}: VAT 15% & 5% present",
                                "15%" in text and "5%" in text))
            elif schema["_meta"]["doc"] == "bilingual_restaurant_bill":
                results.append((f"{name}: Bengali item 'বিরিয়ানি' present", "বিরিয়ানি" in text))
                results.append((f"{name}: Bengali 'ধন্যবাদ' present", "ধন্যবাদ" in text))
                results.append((f"{name}: 'GRAND TOTAL' present", "GRAND TOTAL" in text))
                results.append((f"{name}: 'Service Charge (10%)' present",
                                "Service Charge (10%)" in text))
        else:  # chat txt
            raw = path.read_text(encoding="utf-8")
            nonempty = bool(raw.strip())
            lines = raw.splitlines()
            msg_lines = [ln for ln in lines if CHAT_MSG_RE.match(ln)]
            schema_msgs = [m for m in schema["messages"] if m["kind"] in ("message", "media")]
            results.append((f"{name}: file non-empty", nonempty))
            results.append((f"{name}: all prefixed lines parse via regex",
                            len(msg_lines) == len(schema_msgs)))
            results.append((f"{name}: 30-50 messages", 30 <= len(schema_msgs) <= 50))
            results.append((f"{name}: has multi-line message",
                            any(m["multiline"] for m in schema["messages"])))
            results.append((f"{name}: has media lines",
                            any(m["kind"] == "media" for m in schema["messages"])))
            results.append((f"{name}: has system lines",
                            any(m["kind"] == "system" for m in schema["messages"])))
            results.append((f"{name}: contains Bengali chars", has_bengali(raw)))

    print("\n=== VERIFICATION ===")
    for label, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def write_schema(path, schema):
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate 5 synthetic Bangladeshi sample documents (plan §7) "
                    "plus ground-truth schema.json files.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42; deterministic output)")
    args = parser.parse_args(argv)

    print("Loading reportlab...")
    try:
        import reportlab  # noqa: F401
    except ImportError:
        print("  reportlab not found - pip installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "reportlab"],
                       check=True)
        import reportlab  # noqa: F401

    try:
        import uharfbuzz  # noqa: F401
    except ImportError:
        print("  uharfbuzz not found - pip installing (enables Bengali shaping)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uharfbuzz"],
                       check=True)

    print("Registering fonts...")
    fonts = register_fonts()
    print(f"  Bengali font: {fonts['bengali']}")
    print(f"  Latin font:   {fonts['latin']}")

    for d in (SD06, SD07, SD08):
        d.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    generators = [
        ("bkash_statement", generate_bkash, SD06),
        ("nagad_statement", generate_nagad, SD06),
        ("mushak63_invoice", generate_mushak, SD08),
        ("bilingual_restaurant_bill", generate_restaurant, SD08),
        ("bangla_chat", generate_chat, SD07),
    ]
    artifacts = []
    for name, gen, outdir in generators:
        print(f"Generating {name} ...")
        out_path, schema = gen(seed, fonts)
        schema_path = outdir / f"{name}.schema.json"
        write_schema(schema_path, schema)
        if out_path.suffix == ".pdf":
            normalize_pdf_metadata(out_path)
        artifacts.append((name, "pdf" if out_path.suffix == ".pdf" else "txt",
                          out_path, schema))
        print(f"  wrote {out_path.name} ({out_path.stat().st_size} bytes) "
              f"+ {schema_path.name} ({schema_path.stat().st_size} bytes)")

    ok = verify_all(artifacts)

    print("\n=== SUMMARY ===")
    for name, kind, path, schema in artifacts:
        print(f"  {path.relative_to(REPO_ROOT)}  {path.stat().st_size:>8} bytes")
    print(f"\nSeed used: {seed}")
    print("Verification:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
