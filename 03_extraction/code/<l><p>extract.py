# ==========================================================
# export_lp_to_txt.py
# Extract <l> and <p> to per-TCP txt files
# Input:
#   /Users/grace/Desktop/new meta/deep_tcp.xlsx
# XML roots:
#   /Users/grace/Desktop/new meta/tcp_drama/
#   /Users/grace/Desktop/new meta/after seperation/
# Output:
#   /Users/grace/Desktop/new meta/<l><p>extract/{TCP}.txt
#   /Users/grace/Desktop/new meta/<l><p>extract/_export_log.csv
# ==========================================================

import os
import re
import pandas as pd
from bs4 import BeautifulSoup

BASE_DIR = "/Users/grace/Desktop/new meta"
META_XLSX = os.path.join(BASE_DIR, "deep_tcp.xlsx")

XML_DIRS = [
    os.path.join(BASE_DIR, "tcp_drama"),
    os.path.join(BASE_DIR, "after seperation"),
]

OUT_DIR = os.path.join(BASE_DIR, "<l><p>extract")
OUT_LOG = os.path.join(OUT_DIR, "_export_log.csv")

os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def _local_name(tag_name: str) -> str:
    """Return localname for tags like 'tei:l' -> 'l'."""
    if not tag_name:
        return ""
    return tag_name.split(":")[-1].lower()


def build_xml_index(xml_dirs) -> dict:
    """
    Build mapping: tcp_code (filename without .xml) -> full path
    Searches recursively under given directories.
    """
    index = {}
    for d in xml_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.lower().endswith(".xml"):
                    tcp = fn[:-4].strip()
                    index[tcp] = os.path.join(root, fn)
    return index


# Boilerplate / publication-ish phrases you don't want in drama text
# (This filters out <l>/<p> lines that *contain* these strings.)
PUB_KEYWORDS = [
    "text creation partnership",
    "early english books online",
    "eebo-tcp",
    "proquest",
    "creative commons",
    "public domain",
    "phase i",
    "phase 1",
    "ann arbor",
    "oxford",
    "tei",
    "tcp files",
    "keying",
    "markup guidelines",
    "publisher",
    "pubplace",
    "printed at",
    "printed in",
    "reproduction of the original",
    "images scanned from microfilm",
    "transcribed from",
]


def extract_lp_text(xml_path: str) -> str:
    """
    Robust extraction:
    - Works even if XML is fragment (no single root) by wrapping <root>...</root> if needed
    - Does NOT require <text>/<body> structure
    - Drops teiHeader if present
    - Extracts visible text from <l> and <p> in document order
    """
    raw = open(xml_path, "rb").read()

    # First parse attempt
    soup = BeautifulSoup(raw, "xml")

    # If parsing yields no <l>/<p>, try wrapping as fragment XML
    lp_tags = soup.find_all(lambda t: _local_name(getattr(t, "name", "")) in ("l", "p"))
    if not lp_tags:
        soup = BeautifulSoup(b"<root>" + raw + b"</root>", "xml")
        lp_tags = soup.find_all(lambda t: _local_name(getattr(t, "name", "")) in ("l", "p"))

    # Remove teiHeader if present (prevents accidental header <p>)
    tei_header = soup.find(lambda t: _local_name(getattr(t, "name", "")) == "teiheader")
    if tei_header:
        tei_header.decompose()

    parts = []
    for t in lp_tags:
        s = t.get_text(" ", strip=True)
        if not s:
            continue

        # normalize whitespace
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            continue

        low = s.lower()
        # filter publication/boilerplate-ish lines
        if any(k in low for k in PUB_KEYWORDS):
            continue

        parts.append(s)

    return "\n".join(parts).strip()


def find_col(df: pd.DataFrame, candidates):
    """Case-insensitive column finder."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


# -----------------------------
# Main
# -----------------------------
print("🔎 Loading metadata:", META_XLSX)
df = pd.read_excel(META_XLSX)
print("✅ Loaded. Shape:", df.shape)

col_tcp = find_col(df, ["TCP", "tcp"])
col_rt = find_col(df, ["record_type", "Record_Type", "record type"])

if not col_tcp or not col_rt:
    raise ValueError(f"Missing required columns. Found TCP={col_tcp}, record_type={col_rt}. "
                     f"Columns are: {list(df.columns)}")

# Filter out Collection
df_work = df[df[col_rt].astype(str).str.strip() != "Collection"].copy()
df_work[col_tcp] = df_work[col_tcp].astype(str).str.strip()

print("✅ After filtering Collection:", df_work.shape)

# Index XML files
xml_index = build_xml_index(XML_DIRS)
print(f"✅ XML indexed: {len(xml_index)} files")

log_rows = []
written = 0
missing_xml = 0
empty_text = 0

for _, row in df_work.iterrows():
    tcp = row[col_tcp]
    if not tcp or tcp.lower() == "nan":
        continue

    xml_path = xml_index.get(tcp)
    out_txt = os.path.join(OUT_DIR, f"{tcp}.txt")

    if not xml_path:
        missing_xml += 1
        log_rows.append({
            "TCP": tcp,
            "status": "missing_xml",
            "xml_path": "",
            "txt_path": out_txt,
            "extracted_chars": 0
        })
        continue

    try:
        text = extract_lp_text(xml_path)
    except Exception as e:
        log_rows.append({
            "TCP": tcp,
            "status": f"error: {type(e).__name__}",
            "xml_path": xml_path,
            "txt_path": out_txt,
            "extracted_chars": 0
        })
        continue

    if not text:
        empty_text += 1
        # still write empty file so pipeline is deterministic
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("")
        log_rows.append({
            "TCP": tcp,
            "status": "extracted_empty",
            "xml_path": xml_path,
            "txt_path": out_txt,
            "extracted_chars": 0
        })
        continue

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    written += 1
    log_rows.append({
        "TCP": tcp,
        "status": "ok",
        "xml_path": xml_path,
        "txt_path": out_txt,
        "extracted_chars": len(text)
    })

print("✅ Written:", written)
print("⚠ Missing XML:", missing_xml)
print("⚠ Extracted empty:", empty_text)

log_df = pd.DataFrame(log_rows)
log_df.to_csv(OUT_LOG, index=False)
print("🧾 Log saved:", OUT_LOG)
print("🎉 Done.")