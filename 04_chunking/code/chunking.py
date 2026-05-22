# ==========================================================
# build_chunks_from_txt_corpus.py
# Chunk ONLY (no embedding)
# Input:
#   - metadata: /Users/grace/Desktop/new meta/deep_tcp.xlsx
#   - texts:    /Users/grace/Desktop/new meta/<l><p>extract/*.txt   (named by TCP)
# Output:
#   - /Users/grace/Desktop/new meta/chunking/chunks_chunk2000.csv
#   - /Users/grace/Desktop/new meta/chunking/chunk_ids.csv
# ==========================================================

import os
import re
import pandas as pd

# ----------------------------
# 0) Paths
# ----------------------------
BASE_DIR = "/Users/grace/Desktop/new meta"
META_PATH = os.path.join(BASE_DIR, "deep_tcp.xlsx")

# NOTE: your folder name contains < and > — that's fine for macOS paths
TEXT_DIR = os.path.join(BASE_DIR, "<l><p>extract")

OUT_DIR = os.path.join(BASE_DIR, "chunking")
os.makedirs(OUT_DIR, exist_ok=True)

CHUNK_SIZE = 2000

OUT_CHUNKS_CSV = os.path.join(OUT_DIR, f"chunks_chunk{CHUNK_SIZE}.csv")
OUT_CHUNK_IDS  = os.path.join(OUT_DIR, "chunk_ids.csv")

# ----------------------------
# 1) Chunking function (same spirit as your old code)
# ----------------------------
def chunk_text(text: str, chunk_size: int = 1000):
    """
    Split text into chunks of up to chunk_size characters,
    trying not to split mid-sentence (., ?, !).
    """
    def is_end_of_sentence(ch: str) -> bool:
        return ch in ".!?;"

    text = text or ""

    # Normalize whitespace so we chunk by continuous text, not by original line breaks.
    # This is important because your extracted <l>/<p> corpus is often one line per verse/prose line.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    current = ""

    for ch in text:
        if len(current) + 1 <= chunk_size:
            current += ch
        else:
            # try to cut at the last sentence boundary inside current
            if current and not is_end_of_sentence(current[-1]):
                last_end = max(
                    current.rfind(". "),
                    current.rfind("? "),
                    current.rfind("! "),
                    current.rfind("; ")
                )
                if last_end != -1:
                    # include punctuation
                    cut = current[: last_end + 1].strip()
                    rest = current[last_end + 1 :].strip()
                    if cut:
                        chunks.append(cut)
                    current = (rest + " " + ch).strip()
                else:
                    chunks.append(current.strip())
                    current = ch
            else:
                chunks.append(current.strip())
                current = ch

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ----------------------------
# 2) Load metadata
# ----------------------------
print("🔎 Loading metadata:", META_PATH)
df = pd.read_excel(META_PATH)
print("✅ Loaded. Shape:", df.shape)

# Required columns
required_cols = ["record_type", "TCP"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in deep_tcp.xlsx: {missing}")

# Filter out Collection
df = df[df["record_type"] != "Collection"].copy()
print("✅ After filtering Collection:", df.shape)

# Columns you said you want for chunk table (use exactly these names)
META_COLS_TO_KEEP = [
    "TCP",
    "title.1",
    "author.1",
    "genre_brit_filter",
    "date_first_performance_brit_filter",
]

# Some files might not have all of them; keep what exists
existing_meta_cols = [c for c in META_COLS_TO_KEEP if c in df.columns]
missing_meta_cols = [c for c in META_COLS_TO_KEEP if c not in df.columns]
if missing_meta_cols:
    print("⚠ These metadata columns are not in the sheet (will be skipped):", missing_meta_cols)

# ----------------------------
# 3) Build chunks
# ----------------------------
rows = []
missing_txt = []
empty_txt = 0

# Normalize TCP values to string keys
def norm_tcp(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return s

for _, r in df.iterrows():
    tcp = norm_tcp(r["TCP"])
    if not tcp:
        continue

    txt_path = os.path.join(TEXT_DIR, f"{tcp}.txt")
    if not os.path.exists(txt_path):
        missing_txt.append(tcp)
        continue

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    if not text or not text.strip():
        empty_txt += 1
        continue

    chunks = chunk_text(text, CHUNK_SIZE)
    if not chunks:
        empty_txt += 1
        continue

    # Prepare metadata dict (only selected columns)
    meta_payload = {c: r[c] for c in existing_meta_cols}

    for i, ch in enumerate(chunks):
        ch = ch.strip()
        if not ch:
            continue

        # stable, simple chunk id
        chunk_id = f"{tcp}_{i:05d}"

        row = {
            "TCP": tcp,
            "chunk_id": chunk_id,
            "chunk_index": i,
            "chunk_len": len(ch),
            "chunk_text": ch,
        }
        row.update(meta_payload)
        rows.append(row)

print("✅ Total chunks:", len(rows))
print("⚠ Missing txt files:", len(missing_txt))
print("⚠ Empty txt files:", empty_txt)

# Save missing list for debugging (optional but useful)
if missing_txt:
    missing_path = os.path.join(OUT_DIR, "missing_txt_files.csv")
    pd.DataFrame({"TCP": missing_txt}).to_csv(missing_path, index=False)
    print("🧾 Missing list saved:", missing_path)

# ----------------------------
# 4) Save outputs
# ----------------------------
chunks_df = pd.DataFrame(rows)
chunks_df.to_csv(OUT_CHUNKS_CSV, index=False, encoding="utf-8-sig")
print("✅ Saved chunk table:", OUT_CHUNKS_CSV)

pd.DataFrame({"chunk_id": chunks_df["chunk_id"]}).to_csv(OUT_CHUNK_IDS, index=False, encoding="utf-8-sig")
print("✅ Saved chunk IDs:", OUT_CHUNK_IDS)

print("🎉 Done.")