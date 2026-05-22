"""csv_maker.py

Build Shakespeare vs Non-Shakespeare genre–topic distribution tables
from the chunk-level master table, using the SAME priority-based genre
normalization logic as your full-corpus `genre_topic.py`.

Input (default):
  /Users/grace/Desktop/new meta/final chunk-level master table/chunk_level_master_table.csv

Required columns:
  - genre_brit_filter
  - topic
  - an author column (one of: 'author.1', 'Author_HK', 'author', 'author_name')

Outputs (saved to):
  /Users/grace/Desktop/new meta/non shakespeare_shakespeare/
    Shakespeare/
      genre_topic_counts_long.csv
      genre_topic_pivot_percent.csv
    Non-Shakespeare/
      genre_topic_counts_long.csv
      genre_topic_pivot_percent.csv

Rule:
  - Binary split only: if author string contains 'Shakespeare' (case-insensitive) -> Shakespeare
    else -> Non-Shakespeare (so Anonymous naturally falls into Non-Shakespeare).

Genre normalization:
  - Priority: tragedy > tragicomedy > comedy > history
  - Semicolon/comma/slash tokenization with light normalization of 'tragic-comedy' variants.
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
OUT_ROOT = BASE_DIR / "non shakespeare_shakespeare"


# ----------------------------
# Genre mapping (priority-based) — identical design to your full-corpus script
# ----------------------------
PRIORITY = [
    "tragedy",       # highest
    "tragicomedy",
    "comedy",
    "history",       # lower than tragedy
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


def detect_author_column(df: pd.DataFrame) -> str:
    """Detect an author column name from common candidates."""
    candidates = ["author.1", "Author_HK", "author", "author_name", "Author", "author_hk"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "No author column found. Expected one of: "
        + ", ".join(candidates)
        + f". Found columns: {list(df.columns)}"
    )


def assign_group(author_val: object) -> str:
    """Binary split: contains Shakespeare -> Shakespeare else Non-Shakespeare."""
    if author_val is None or (isinstance(author_val, float) and pd.isna(author_val)):
        return "Non-Shakespeare"
    s = str(author_val).strip()
    if not s:
        return "Non-Shakespeare"
    return "Shakespeare" if "shakespeare" in s.lower() else "Non-Shakespeare"


def build_tables(df: pd.DataFrame, out_dir: Path) -> None:
    """Build and save A) long and B) pivot tables for a subset df."""
    out_dir.mkdir(parents=True, exist_ok=True)

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

    grouped["Genre_Main"] = pd.Categorical(
        grouped["Genre_Main"],
        categories=["Comedy", "Tragedy", "History"],
        ordered=True,
    )
    grouped = grouped.sort_values(["Genre_Main", "ChunkCount"], ascending=[True, False]).reset_index(drop=True)
    grouped["Genre_Main"] = grouped["Genre_Main"].astype(str)

    grouped.to_csv(out_dir / "genre_topic_counts_long.csv", index=False, encoding="utf-8-sig")

    # ----------------------------
    # B) Pivot table: Genre_Main x topic (within-genre %)
    # ----------------------------
    pivot_counts = pd.crosstab(df["Genre_Main"], df["topic"]).astype(int)

    num_chunks = pivot_counts.sum(axis=1)
    pivot_percent = pivot_counts.div(num_chunks, axis=0) * 100

    pivot_percent.insert(0, "NumChunks", num_chunks.astype(int))
    pivot_percent = pivot_percent.reindex(["Comedy", "Tragedy", "History"])
    pivot_percent = pivot_percent.round(4)

    pivot_percent.to_csv(out_dir / "genre_topic_pivot_percent.csv", encoding="utf-8-sig")


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {IN_CSV}")

    print(f"🔎 Reading: {IN_CSV}")
    df = pd.read_csv(IN_CSV)
    print(f"✅ Loaded df: {df.shape}")

    # Validate required columns
    required = ["genre_brit_filter", "topic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in master table: {missing}")

    author_col = detect_author_column(df)
    print(f"✅ Using author column: {author_col}")

    # Drop outlier topic if present (BERTopic uses -1 for outliers)
    df = df.copy()
    df["topic"] = pd.to_numeric(df["topic"], errors="coerce")
    before = len(df)
    df = df[df["topic"].notna()].copy()
    df["topic"] = df["topic"].astype(int)
    df = df[df["topic"] != -1].copy()
    print(f"✅ Kept {len(df)} / {before} rows after dropping topic==-1 and NaNs")

    # Binary group split
    df["Group"] = df[author_col].apply(assign_group)

    # Build Genre_Main using priority mapping
    df["Genre_Main"] = df["genre_brit_filter"].apply(genre_main_priority)

    # Restrict to the three analyzed genres only (comparative design)
    keep_genres = {"Tragedy", "Comedy", "History"}
    before2 = len(df)
    df = df[df["Genre_Main"].isin(keep_genres)].copy()
    print(f"✅ Kept {len(df)} / {before2} rows after restricting to 3 genres")

    # Save quick diagnostics
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (
        df["Group"].value_counts().rename_axis("Group").reset_index(name="Rows")
        .to_csv(OUT_ROOT / "debug_group_counts.csv", index=False, encoding="utf-8-sig")
    )
    (
        df.groupby(["Group", "Genre_Main"], as_index=False)
        .size()
        .rename(columns={"size": "Rows"})
        .sort_values(["Group", "Rows"], ascending=[True, False])
        .to_csv(OUT_ROOT / "debug_genres_by_group.csv", index=False, encoding="utf-8-sig")
    )

    # Split and build tables
    df_sh = df[df["Group"] == "Shakespeare"].copy()
    df_non = df[df["Group"] == "Non-Shakespeare"].copy()

    print(f"📊 Shakespeare rows: {len(df_sh)}")
    print(f"📊 Non-Shakespeare rows: {len(df_non)}")

    build_tables(df_sh, OUT_ROOT / "Shakespeare")
    build_tables(df_non, OUT_ROOT / "Non-Shakespeare")

    print("🎉 Done.")
    print("Shakespeare outputs:", OUT_ROOT / "Shakespeare")
    print("Non-Shakespeare outputs:", OUT_ROOT / "Non-Shakespeare")


if __name__ == "__main__":
    main()