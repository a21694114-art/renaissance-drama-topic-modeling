# topic_modeling_keywords.py
# Purpose:
#   - Load precomputed embeddings + chunk table + baseline cluster table
#   - Keep all baseline clusters for BERTopic modeling
#   - Lightly clean text and apply the existing stopword-removal workflow
#   - Fit BERTopic with embeddings using the SAME baseline UMAP/HDBSCAN parameters
#   - Update topic keywords using KeyBERTInspired + MMR
#   - Export keyword tables (top6/top8/top10/top20) + topic_info + representative docs + chunk_topics_filtered

import os
import re
import numpy as np
import pandas as pd
import plotly.express as px

from sentence_transformers import SentenceTransformer

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer

from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance


# =========================
# 0) Paths
# =========================
OUT_DIR = "/Users/grace/Desktop/new meta/topic_modeling_keywords"
os.makedirs(OUT_DIR, exist_ok=True)


EMB_PATH = "/Users/grace/Desktop/new meta/embedding/embeddings_gte_chunk2000.npy"
CHUNKS_PATH = "/Users/grace/Desktop/new meta/chunking/chunks_chunk2000.csv"
CLUSTER_PATH = "/Users/grace/Desktop/new meta/clustering/cluster_xy_table__baseline.csv"

# keyword export settings
TOP_N_LIST = [6, 8, 10, 20]


# =========================
# 1) Load inputs
# =========================
print(f"🔎 Loading embeddings: {EMB_PATH}")
embeddings = np.load(EMB_PATH)
print("✅ Embeddings shape:", embeddings.shape)

print(f"🔎 Loading chunks: {CHUNKS_PATH}")
chunks_df = pd.read_csv(CHUNKS_PATH)
print("✅ chunks_df shape:", chunks_df.shape)

print(f"🔎 Loading clusters: {CLUSTER_PATH}")
cluster_df = pd.read_csv(CLUSTER_PATH)
print("✅ cluster_df shape:", cluster_df.shape)

# Validate row alignment
if len(chunks_df) != len(cluster_df) or len(chunks_df) != embeddings.shape[0]:
    raise ValueError(
        f"Row mismatch: chunks_df={len(chunks_df)}, cluster_df={len(cluster_df)}, embeddings={embeddings.shape[0]}"
    )

# Detect the text column
TEXT_COL_CANDIDATES = ["chunk_text", "text_chunk", "Text", "text"]
text_col = None
for c in TEXT_COL_CANDIDATES:
    if c in chunks_df.columns:
        text_col = c
        break
if text_col is None:
    raise ValueError(f"Cannot find a text column in chunks_df. Tried: {TEXT_COL_CANDIDATES}")
print(f"✅ Using text column: {text_col}")

# Detect the cluster label column
CLUSTER_COL_CANDIDATES = ["cluster", "Cluster", "cluster_label", "labels"]
cluster_col = None
for c in CLUSTER_COL_CANDIDATES:
    if c in cluster_df.columns:
        cluster_col = c
        break
if cluster_col is None:
    raise ValueError(f"Cannot find a cluster column in cluster_df. Tried: {CLUSTER_COL_CANDIDATES}")
print(f"✅ Using cluster column: {cluster_col}")

cluster_labels = cluster_df[cluster_col].astype(str)


# =========================
# 2) Keep all chunks for BERTopic
# =========================
texts_raw = chunks_df[text_col].astype(str).tolist()
embeddings_keep = embeddings.copy()
kept_idx = np.arange(len(chunks_df))

print(f"✅ Keeping all {len(texts_raw)} chunks for BERTopic modeling.")


# =========================
# 3) Basic cleaning
# =========================
def basic_clean(s: str) -> str:
    # keep as light as possible; do NOT aggressively normalize spelling
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"\s+", " ", s).strip()
    return s

texts = [basic_clean(t) for t in texts_raw]

# drop empty after basic clean
nonempty_mask = np.array([len(t) > 0 for t in texts], dtype=bool)
texts = [t for t in texts if t]
embeddings_keep = embeddings_keep[nonempty_mask]
kept_idx = kept_idx[nonempty_mask]  # keep alignment for saving later

print(f"✅ After basic_clean, kept {len(texts)} texts")


# =========================
# 4) Early Modern stopwords + remove_stopwords
# =========================
# Note: Do NOT include thematic words like love/loue/lord/lorde in stopwords.
stopwords = set(ENGLISH_STOP_WORDS).union({
    # Archaic and Shakespearean function words
    "thou", "thee", "thy", "thine", "ye", "hath", "doth", "art", "shalt", "hast", "didst", "wilt",
    "shall", "would", "could", "might", "must", "unto", "tis", "’tis",

    # Early Modern English variants
    "haue", "doe", "ile", "ll", "vs", "wil", "hee", "saye", "nowe", "maye", "theyr", "dyd",
    "whiche", "mee", "vnto", "don",

    # Modern high-frequency stopwords (already in ENGLISH_STOP_WORDS, but keep explicit)
    "the", "and", "you", "your", "me", "he", "she", "him", "her", "my", "not", "but", "be", "to",
    "of", "in", "for", "on", "with",

    # Dialogic filler / weak verbs / modal-like
    "come", "good", "man", "did", "make", "let", "like", "owe", "now", "yet", "thus", "again",
    "may", "will", "go", "goe", "goeth", "came", "cometh", "yes", "yea", "yeay", "nay",

    # Speech / cognition verbs
    "say", "sayst", "know", "knowest", "speak", "speaketh", "speake", "think", "thinkest", "thinke",

    # Other non-informative or obsolete forms
    "sir", "hys", "thys", "wyll", "yf", "hym", "suche", "nat", "wolde", "thu", "shee", "selfe",
    "le", "em", "se", "thynge",

    # Newly added based on inspection
    "soft", "verily", "marry", "troth", "well", "oh", "bee",
    "ha", "ane", "syr", "tyme", "lyke", "whych", "yow", "welth", "lyfe", "wynde", "neuer",

    # OCR / fragment artifacts observed in corpus
    "th", "whil", "whilst", "vpon", "vppon", "vp", "vntill",

    # Very high-frequency dialogic verbs
    "giue", "tell", "told", "telleth",

    # Keep from your earlier list
    "deuyll", "myne", "longe",
})

def remove_stopwords(text: str) -> str:
    # Only alphabetic tokens; early modern spelling stays
    words = re.findall(r"\b[a-z]+\b", text.lower())
    return " ".join([w for w in words if w not in stopwords])

texts_nostop = [remove_stopwords(t) for t in texts]

# drop empty after stopword removal
nonempty_mask2 = np.array([len(t) > 0 for t in texts_nostop], dtype=bool)
texts_nostop = [t for t in texts_nostop if t]
embeddings_keep = embeddings_keep[nonempty_mask2]
kept_idx = kept_idx[nonempty_mask2]

print(f"✅ After stopword removal, kept {len(texts_nostop)} texts for BERTopic")


# =========================
# 5) BERTopic (OLD approach)
# =========================
# Embedding model is required for KeyBERTInspired/MMR in update_topics
# (BERTopic needs an embedding_model to embed representative documents)
EMBEDDING_MODEL_NAME = "thenlper/gte-large"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"✅ Loaded embedding model for representations: {EMBEDDING_MODEL_NAME}")

umap_model = UMAP(
    n_components=5,
    n_neighbors=15,
    min_dist=0.05,
    metric="cosine",
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=30,
    min_samples=None,
    metric="euclidean",
    cluster_selection_method="eom"
)

# We do NOT pass stop_words here (avoids sklearn stop_words validation issue).
# Since texts are already cleaned, vectorizer just builds vocab.
vectorizer_model = CountVectorizer(
    ngram_range=(1, 1),
    min_df=10,
    max_df=0.6,
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=False,
    verbose=True
)

print("🚀 Fitting BERTopic...")
topics, probs = topic_model.fit_transform(texts_nostop, embeddings_keep)


# =========================
# 6) Representation refinement (KeyBERTInspired + MMR)
# =========================
representation_model = {
    "KeyBERT": KeyBERTInspired(),
    "MMR": MaximalMarginalRelevance(diversity=0.3)
}
topic_model.update_topics(texts_nostop, representation_model=representation_model)


# =========================
# 7) Export outputs (NORMAL filenames)
# =========================
# 7.1 Topic info
topic_info = topic_model.get_topic_info()
topic_info.to_csv(os.path.join(OUT_DIR, "topic_info.csv"), index=False)

# 7.2 Top-N keywords per topic (multiple N)
rows = []
for t in topic_info["Topic"].tolist():
    if int(t) == -1:
        continue
    words = topic_model.get_topic(int(t))
    if not words:
        continue
    full_words = [w for w, _ in words]
    count_val = int(topic_info.loc[topic_info["Topic"] == t, "Count"].values[0])
    rows.append({"topic": int(t), "count": count_val, "all_words": full_words})

base_df = pd.DataFrame(rows).sort_values(["count"], ascending=False).reset_index(drop=True)

for TOP_N in TOP_N_LIST:
    out_rows = []
    for _, r in base_df.iterrows():
        top_words = r["all_words"][:TOP_N]
        out_rows.append({
            "topic": r["topic"],
            "count": r["count"],
            "top_words": ", ".join(top_words)
        })
    keywords_df = pd.DataFrame(out_rows).sort_values(["count"], ascending=False)
    keywords_df.to_csv(os.path.join(OUT_DIR, f"topic_keywords_top{TOP_N}.csv"), index=False)

# 7.3 Representative docs
rep_docs = topic_model.get_representative_docs()
rep_rows = []
for t, docs in rep_docs.items():
    if int(t) == -1:
        continue
    for i, d in enumerate(docs[:5]):
        rep_rows.append({"topic": int(t), "rank": i + 1, "doc_snippet": str(d)[:600]})
rep_df = pd.DataFrame(rep_rows)
rep_df.to_csv(os.path.join(OUT_DIR, "topic_representative_docs.csv"), index=False)

# 7.4 Per-chunk topic assignment (all chunks retained)
assign_df = chunks_df.loc[kept_idx].copy()
assign_df["topic"] = topics
assign_df.to_csv(os.path.join(OUT_DIR, "chunk_topics_filtered.csv"), index=False)

# 7.5 Save model folder
model_path = os.path.join(OUT_DIR, "bertopic_model")
# NOTE: if this folder already exists, BERTopic may refuse to overwrite.
# You can delete bertopic_model/ before re-running.
topic_model.save(model_path)


# =========================
# 8) Console preview
# =========================
TOP_PREVIEW = 8
preview_path = os.path.join(OUT_DIR, f"topic_keywords_top{TOP_PREVIEW}.csv")
if os.path.exists(preview_path):
    preview_df = pd.read_csv(preview_path)
    print(f"\n📌 Quick preview: Top 15 topics by size (TOP {TOP_PREVIEW} words)")
    print(preview_df.head(15).to_string(index=False))

print("\n✅ Saved outputs to:", OUT_DIR)
print(" - topic_info.csv")
print(" - topic_keywords_top6.csv")
print(" - topic_keywords_top8.csv")
print(" - topic_keywords_top10.csv")
print(" - topic_keywords_top20.csv")
print(" - topic_representative_docs.csv")
print(" - chunk_topics_filtered.csv")
print(" - bertopic_model/")
print("\n🎉 Topic modeling completed.")