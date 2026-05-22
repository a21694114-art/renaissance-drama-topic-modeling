

#!/usr/bin/env python3
"""Build a final chunk-level master table.

Inputs:
  1) chunk_topics_filtered.csv
  2) cluster_xy_table__baseline.csv
  3) topic_summaries.csv

Output:
  - chunk_level_master_table.csv
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import pandas as pd


def _read_csv_safely(path: str) -> pd.DataFrame:
    """Read CSV robustly across common encodings used in this project."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    encodings = ["utf-8-sig", "utf-8", "cp949", "latin1"]
    last_err: Optional[Exception] = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e

    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read CSV with common encodings: {path}. "
            f"Last errors: {last_err} / {e}"
        )


def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_topic_int(series: pd.Series) -> pd.Series:
    def to_int(v):
        if pd.isna(v):
            return pd.NA
        s = str(v).strip()
        if s == "":
            return pd.NA
        try:
            return int(float(s))
        except Exception:
            return pd.NA

    return series.apply(to_int).astype("Int64")


def _ensure_key(df: pd.DataFrame, preferred: str, fallbacks: List[str]) -> Tuple[pd.DataFrame, str]:
    """Ensure df has a merge key column; return (df, key_col_name)."""
    if preferred in df.columns:
        return df, preferred
    for fb in fallbacks:
        if fb in df.columns:
            return df.rename(columns={fb: preferred}), preferred
    raise KeyError(
        f"Could not find a key column. Expected '{preferred}' or one of {fallbacks}. "
        f"Existing columns: {list(df.columns)}"
    )


def build_master_table(
    chunks_csv: str,
    xy_csv: str,
    topic_summary_csv: str,
    out_dir: str,
    drop_topic_minus1: bool = True,
) -> str:
    print(f"🔎 Loading chunks table: {chunks_csv}")
    chunks_df = _read_csv_safely(chunks_csv)
    print(f"✅ chunks_df shape: {chunks_df.shape}")

    print(f"🔎 Loading XY table: {xy_csv}")
    xy_df = _read_csv_safely(xy_csv)
    print(f"✅ xy_df shape: {xy_df.shape}")

    print(f"🔎 Loading topic summary table: {topic_summary_csv}")
    ts_df = _read_csv_safely(topic_summary_csv)
    print(f"✅ topic_summaries_df shape: {ts_df.shape}")

    # Normalize keys
    chunks_df, _ = _ensure_key(
        chunks_df,
        preferred="chunk_id",
        fallbacks=["hk_id", "chunkid", "Chunk_ID"],
    )
    xy_df, _ = _ensure_key(
        xy_df,
        preferred="chunk_id",
        fallbacks=["hk_id", "chunkid", "Chunk_ID"],
    )

    # Normalize topic in chunks
    topic_col = _pick_first_existing(chunks_df, ["topic", "Topic"])
    if topic_col is None:
        raise KeyError(f"No topic column found in chunks table. Columns: {list(chunks_df.columns)}")
    if topic_col != "topic":
        chunks_df = chunks_df.rename(columns={topic_col: "topic"})
    chunks_df["topic"] = _coerce_topic_int(chunks_df["topic"])

    # Normalize x/y in XY table
    x_col = _pick_first_existing(xy_df, ["x", "X"])
    y_col = _pick_first_existing(xy_df, ["y", "Y"])
    if x_col is None or y_col is None:
        raise KeyError(f"XY table must contain x and y columns. Columns: {list(xy_df.columns)}")
    if x_col != "x":
        xy_df = xy_df.rename(columns={x_col: "x"})
    if y_col != "y":
        xy_df = xy_df.rename(columns={y_col: "y"})

    # Normalize cluster column names
    cluster_chunks_col = _pick_first_existing(chunks_df, ["cluster", "Cluster"])
    if cluster_chunks_col and cluster_chunks_col != "cluster":
        chunks_df = chunks_df.rename(columns={cluster_chunks_col: "cluster"})

    cluster_xy_col = _pick_first_existing(xy_df, ["cluster", "Cluster"])
    if cluster_xy_col and cluster_xy_col != "cluster":
        xy_df = xy_df.rename(columns={cluster_xy_col: "cluster"})

    # Prepare XY merge frame
    xy_keep_cols = ["chunk_id", "x", "y"]
    if "cluster" in xy_df.columns:
        xy_keep_cols.append("cluster")
    xy_merge = xy_df[xy_keep_cols].copy()

    if xy_merge.duplicated("chunk_id").any():
        dup_n = int(xy_merge.duplicated("chunk_id").sum())
        print(f"⚠ XY table has {dup_n} duplicate chunk_id rows. Keeping first occurrence.")
        xy_merge = xy_merge.drop_duplicates("chunk_id", keep="first")

    # Merge chunks + xy
    master = chunks_df.merge(xy_merge, on="chunk_id", how="left", suffixes=("", "_xy"))

    # Fill cluster from xy if missing in chunks
    if "cluster" in master.columns and "cluster_xy" in master.columns:
        master["cluster"] = master["cluster"].fillna(master["cluster_xy"])
        master = master.drop(columns=["cluster_xy"])

    # Normalize topic in topic summary table
    ts_topic_col = _pick_first_existing(ts_df, ["topic", "Topic"])
    if ts_topic_col is None:
        raise KeyError(f"topic_summaries.csv must have a 'topic' column. Columns: {list(ts_df.columns)}")
    if ts_topic_col != "topic":
        ts_df = ts_df.rename(columns={ts_topic_col: "topic"})
    ts_df["topic"] = _coerce_topic_int(ts_df["topic"])

    # Normalize topic summary column
    summary_col = _pick_first_existing(ts_df, ["topic_summary", "summary", "label"])
    if summary_col is None:
        raise KeyError(
            f"topic_summaries.csv must have a topic summary column "
            f"(e.g. 'topic_summary'). Columns: {list(ts_df.columns)}"
        )
    if summary_col != "topic_summary":
        ts_df = ts_df.rename(columns={summary_col: "topic_summary"})

    # Normalize top_words if present
    top_words_col = _pick_first_existing(
        ts_df,
        ["top_words", "topwords", "keywords", "topic_keywords"],
    )
    if top_words_col and top_words_col != "top_words":
        ts_df = ts_df.rename(columns={top_words_col: "top_words"})

    # Select optional extra columns
    extra_cols: List[str] = []
    if "top_words" in ts_df.columns:
        extra_cols.append("top_words")
    if "count" in ts_df.columns:
        extra_cols.append("count")

    ts_merge = ts_df[["topic", "topic_summary"] + extra_cols].drop_duplicates("topic")

    # Merge topic summaries
    master = master.merge(ts_merge, on="topic", how="left")

    # Build topic_label
    def _make_topic_label(t, s):
        if pd.isna(t):
            return pd.NA
        if pd.isna(s) or str(s).strip() == "":
            return str(int(t))
        return f"{int(t)}. {str(s).strip()}"

    master["topic_label"] = [
        _make_topic_label(t, s)
        for t, s in zip(master["topic"].tolist(), master["topic_summary"].tolist())
    ]

    # Optional: drop topic == -1
    if drop_topic_minus1:
        before = len(master)
        master = master[master["topic"].fillna(-9999) != -1].copy()
        after = len(master)
        print(f"🧹 Dropped topic == -1 rows: {before - after}")

    # Diagnostics
    missing_xy = int(master["x"].isna().sum()) if "x" in master.columns else -1
    missing_summary = int(master["topic_summary"].isna().sum()) if "topic_summary" in master.columns else -1
    unique_topics = master["topic"].nunique(dropna=True) if "topic" in master.columns else -1

    print(f"📊 Unique topics in master: {unique_topics}")
    if missing_xy > 0:
        print(f"⚠ Warning: {missing_xy} rows missing x/y after merge (check chunk_id alignment).")
    if missing_summary > 0:
        print(f"⚠ Warning: {missing_summary} rows missing topic_summary after merge (check topic alignment).")

    # Reorder columns
    preferred_order = [
        "TCP",
        "chunk_id",
        "chunk_index",
        "chunk_len",
        "chunk_text",
        "title.1",
        "author.1",
        "genre_brit_filter",
        "date_first_performance_brit_filter",
        "cluster",
        "topic",
        "topic_label",
        "topic_summary",
        "top_words",
        "count",
        "x",
        "y",
    ]
    cols_first = [c for c in preferred_order if c in master.columns]
    cols_rest = [c for c in master.columns if c not in cols_first]
    master = master[cols_first + cols_rest]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "chunk_level_master_table.csv")
    master.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"✅ Master table saved: {out_path}")
    print(f"✅ Final shape: {master.shape}")
    return out_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final chunk-level master table")
    p.add_argument(
        "--chunks_csv",
        default="/Users/grace/Desktop/new meta/topic_modeling_keywords/chunk_topics_filtered.csv",
        help="Path to chunk_topics_filtered.csv",
    )
    p.add_argument(
        "--xy_csv",
        default="/Users/grace/Desktop/new meta/clustering/cluster_xy_table__baseline.csv",
        help="Path to cluster_xy_table__baseline.csv",
    )
    p.add_argument(
        "--topic_summary_csv",
        default="/Users/grace/Desktop/new meta/gpt_topic/topic_summaries.csv",
        help="Path to topic_summaries.csv",
    )
    p.add_argument(
        "--out_dir",
        default="/Users/grace/Desktop/new meta/final chunk-level master table",
        help="Output directory",
    )
    p.add_argument(
        "--drop_topic_minus1",
        action="store_true",
        default=True,
        help="Drop rows where topic == -1 (default: True)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    build_master_table(
        chunks_csv=args.chunks_csv,
        xy_csv=args.xy_csv,
        topic_summary_csv=args.topic_summary_csv,
        out_dir=args.out_dir,
        drop_topic_minus1=args.drop_topic_minus1,
    )


if __name__ == "__main__":
    main()