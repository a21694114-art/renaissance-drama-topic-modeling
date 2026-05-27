"""genre_topic.py

Build genre–topic distribution tables from the chunk-level master table.

Input:
  - /Users/grace/Desktop/new meta/final chunk-level master table/chunk_level_master_table.csv

Key columns used:
  - genre_brit_filter  (raw genre label(s), often semicolon-separated)
  - topic              (topic id)

Outputs (saved to /Users/grace/Desktop/new meta/genre):
  A) genre_topic_counts_long.csv   (Genre_Main x topic, counts + within-genre %)
  B) genre_topic_pivot_percent.csv (Genre_Main x topic pivot, within-genre %)

Genre normalization uses a priority rule (e.g., tragedy > history).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path("/Users/grace/Desktop/new meta")
IN_CSV = BASE_DIR / "final chunk-level master table" / "chunk_level_master_table.csv"
OUT_DIR = BASE_DIR / "genre"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LONG = OUT_DIR / "genre_topic_counts_long.csv"          # A
OUT_PIVOT = OUT_DIR / "genre_topic_pivot_percent.csv"       # B


# ----------------------------
# Genre mapping (priority-based)
# ----------------------------

PRIORITY = [
    "tragedy",       # highest
    "tragicomedy",
    "comedy",
    "history",       # lower than tragedy
]

# Genres retained for the main analysis tables
MAJOR_GENRES = [
    "Comedy",
    "Tragedy",
    "History",
    "Tragicomedy",
]


def _split_genre_tokens(raw: str) -> list[str]:
    """Split a raw genre string into normalized tokens.

    Handles separators like ';' and ',' and normalizes whitespace/case.
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "null", "none listed", "not in britdrama"}:
        return []

    # Normalize separators: semicolon/comma/slash -> ';'
    s = re.sub(r"[,/]+", ";", s)
    parts = [p.strip().lower() for p in s.split(";")]
    parts = [p for p in parts if p]

    # Light normalization for common variants
    normed: list[str] = []
    for p in parts:
        p = p.replace("tragic-comedy", "tragicomedy")
        p = p.replace("tragic comedy", "tragicomedy")
        normed.append(p)

    return normed


def genre_main_priority(raw: str) -> str:
    """Return Genre_Main using priority rule.

    Example: if raw contains both 'tragedy' and 'history' -> 'Tragedy'.
    """
    tokens = _split_genre_tokens(raw)
    if not tokens:
        return "Other"

    # Priority match: substring match within tokens to be robust.
    for key in PRIORITY:
        for t in tokens:
            if key in t:
                return key.capitalize()

    return "Other"


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {IN_CSV}")

    print(f"🔎 Reading: {IN_CSV}")
    df = pd.read_csv(IN_CSV)
    print(f"✅ Loaded df: {df.shape}")

    # Validate columns
    required = ["genre_brit_filter", "topic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in master table: {missing}")

    # Drop outlier topic if present (BERTopic uses -1 for outliers)
    df = df.copy()
    df["topic"] = pd.to_numeric(df["topic"], errors="coerce")
    before = len(df)
    df = df[df["topic"].notna()].copy()
    df["topic"] = df["topic"].astype(int)
    df = df[df["topic"] != -1].copy()
    print(f"✅ Kept {len(df)} / {before} rows after dropping topic==-1 and NaNs")

    # Build Genre_Main
    df["Genre_Main"] = df["genre_brit_filter"].apply(genre_main_priority)

    # Keep only the four major genres used in the analysis
    before_genre_filter = len(df)
    df = df[df["Genre_Main"].isin(MAJOR_GENRES)].copy()
    print(f"✅ Kept {len(df)} / {before_genre_filter} rows after filtering to the four major genres")

    # Stable display/order for tables
    df["Genre_Main"] = pd.Categorical(
        df["Genre_Main"],
        categories=MAJOR_GENRES,
        ordered=True,
    )

    # Optional: attach stable topic label/summary if present
    topic_meta_cols: list[str] = []
    for c in ["topic_label", "topic_summary", "top_words", "topic_keywords", "topic_keywords_top8"]:
        if c in df.columns:
            topic_meta_cols.append(c)

    topic_meta = None
    if topic_meta_cols:
        topic_meta = (
            df[["topic"] + topic_meta_cols]
            .drop_duplicates(subset=["topic"], keep="first")
        )

    # ----------------------------
    # A) Long table: Genre_Main x topic
    # ----------------------------
    grouped = (
        df.groupby(["Genre_Main", "topic"], as_index=False)
        .size()
        .rename(columns={"size": "ChunkCount"})
    )

    grouped["NumChunks_in_Genre"] = grouped.groupby("Genre_Main")["ChunkCount"].transform("sum")
    grouped["Percentage_within_Genre"] = (grouped["ChunkCount"] / grouped["NumChunks_in_Genre"] * 100).round(4)

    if topic_meta is not None and not topic_meta.empty:
        grouped = grouped.merge(topic_meta, on="topic", how="left")

    grouped = grouped.sort_values(["Genre_Main", "ChunkCount"], ascending=[True, False]).reset_index(drop=True)
    grouped["Genre_Main"] = grouped["Genre_Main"].astype(str)
    grouped.to_csv(OUT_LONG, index=False, encoding="utf-8-sig")
    print(f"💾 Saved A (long): {OUT_LONG}")

    # ----------------------------
    # B) Pivot table: Genre_Main x topic (within-genre %)
    # ----------------------------
    # Use crosstab to avoid the 'topic not 1-dimensional' error (topic is used only as columns).
    pivot_counts = pd.crosstab(df["Genre_Main"], df["topic"]).astype(int)

    num_chunks = pivot_counts.sum(axis=1)
    pivot_percent = pivot_counts.div(num_chunks, axis=0) * 100

    pivot_percent.insert(0, "NumChunks", num_chunks.astype(int))
    pivot_percent = pivot_percent.reindex(MAJOR_GENRES)
    pivot_percent = pivot_percent.round(4)

    pivot_percent.to_csv(OUT_PIVOT, encoding="utf-8-sig")
    print(f"💾 Saved B (pivot %): {OUT_PIVOT}")

    # Quick sanity check
    print("\n📌 Sanity check: Percentage rows (excluding NumChunks) should sum to ~100")
    check = pivot_percent.drop(columns=["NumChunks"]).fillna(0).sum(axis=1).round(4)
    print(check)

    print("\n🎉 Done.")


if __name__ == "__main__":
    main()