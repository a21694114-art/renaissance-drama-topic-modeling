# ==========================================================
# embed_chunks_gte.py
# Build embeddings from chunk table (CSV) and save as .npy
#
# Input:
#   /Users/grace/Desktop/new meta/chunking/chunks_chunk2000.csv
# Output:
#   /Users/grace/Desktop/new meta/chunking/embeddings_gte_chunk2000.npy
#   /Users/grace/Desktop/new meta/chunking/embeddings_gte_chunk2000_shape.txt
#   /Users/grace/Desktop/new meta/chunking/embeddings_chunk_ids.csv   (alignment check)
# ==========================================================

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ----------------------------
# 0) Paths / params
# ----------------------------
PROJECT_ROOT = "/Users/grace/Desktop/new meta"
CHUNK_DIR = os.path.join(PROJECT_ROOT, "chunking")
EMBED_DIR = os.path.join(PROJECT_ROOT, "embedding")

os.makedirs(EMBED_DIR, exist_ok=True)

CHUNKS_CSV = os.path.join(CHUNK_DIR, "chunks_chunk2000.csv")

OUT_NPY = os.path.join(EMBED_DIR, "embeddings_gte_chunk2000.npy")
OUT_SHAPE_TXT = os.path.join(EMBED_DIR, "embeddings_gte_chunk2000_shape.txt")
OUT_IDS = os.path.join(EMBED_DIR, "embeddings_chunk_ids.csv")

MODEL_NAME = "thenlper/gte-large"
BATCH_SIZE = 16
NORMALIZE = False  # keep False unless you explicitly want cosine-ready vectors

# ----------------------------
# 1) Load chunks
# ----------------------------
print("🔎 Loading chunks:", CHUNKS_CSV)
df = pd.read_csv(CHUNKS_CSV, encoding="utf-8-sig")
print("✅ Loaded. Shape:", df.shape)
print("📌 Columns:", list(df.columns))

required = ["chunk_id", "chunk_text"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in chunks CSV: {missing}")

# Ensure strings
df["chunk_text"] = df["chunk_text"].astype(str)
df["chunk_id"] = df["chunk_id"].astype(str)

# Drop truly empty chunks (should be rare)
mask = df["chunk_text"].str.strip() != ""
df = df[mask].copy()
df.reset_index(drop=True, inplace=True)

chunk_texts = df["chunk_text"].tolist()
chunk_ids = df["chunk_id"].tolist()

print("✅ Non-empty chunks:", len(chunk_texts))
if len(chunk_texts) == 0:
    raise ValueError("No non-empty chunk_text found.")

# ----------------------------
# 2) Load model
# ----------------------------
print("🔧 Loading model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

# ----------------------------
# 3) Encode embeddings
# ----------------------------
print("🧠 Encoding embeddings...")
embeddings = model.encode(
    chunk_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=NORMALIZE
)

embeddings = np.asarray(embeddings)
print("✅ Embeddings shape:", embeddings.shape)

if embeddings.shape[0] != len(chunk_texts):
    raise RuntimeError("Mismatch: embeddings rows != number of chunk_texts")

# ----------------------------
# 4) Save outputs
# ----------------------------
np.save(OUT_NPY, embeddings)
print("✅ Saved embeddings:", OUT_NPY)

with open(OUT_SHAPE_TXT, "w", encoding="utf-8") as f:
    f.write(f"chunks={len(chunk_texts)}\n")
    f.write(f"dim={embeddings.shape[1]}\n")
print("✅ Saved shape info:", OUT_SHAPE_TXT)

# Save chunk_id alignment list (so you can always map embeddings[i] -> chunk_id)
pd.DataFrame({"chunk_id": chunk_ids}).to_csv(OUT_IDS, index=False, encoding="utf-8-sig")
print("✅ Saved embedding chunk_ids:", OUT_IDS)

print("🎉 Done.")