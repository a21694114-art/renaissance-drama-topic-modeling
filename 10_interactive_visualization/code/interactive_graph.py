"""interactive_graph.py

Generate interactive Plotly topic maps (date-highlight and genre-highlight)
from a pre-merged chunk-level master table.

Expected (typical) columns in the input CSV:
- x, y: 2D coordinates for each chunk
- topic: BERTopic topic id (may include -1 outliers)
- topic_label: e.g., "16. Divine justice and moral reckoning" (preferred)
- topic_summary: short summary phrase (optional)
- title.1 / Title_HK, author.1 / Author_HK, genre_brit_filter / Genre, date_first_performance / Date
- chunk_text: text content for hover

This script is robust to different column naming conventions and will auto-detect
reasonable alternatives.

Usage examples:
  python3 interactive_graph.py --in_csv "/path/to/master_table.csv" --out_dir "/path/to/output"

Outputs:
  - interactive_topics_date_highlight.html
  - interactive_topics_genre_highlight.html
  - interactive_topics_author_highlight.html
  - interactive_topics_title_highlight.html
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ----------------------------
# Helpers
# ----------------------------

def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column name that exists in df from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_int(x) -> Optional[int]:
    """Best-effort conversion to int; returns None on failure."""
    try:
        if pd.isna(x):
            return None
        # handle strings like "1594" or "1594.0"
        s = str(x).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def build_topic_label(df: pd.DataFrame, topic_col: str, topic_label_col: Optional[str], topic_summary_col: Optional[str]) -> pd.Series:
    """Ensure we have a topic_label series.

    Priority:
      1) existing topic_label column
      2) f"{topic}. {topic_summary}" if topic_summary exists
      3) str(topic)
    """
    if topic_label_col and topic_label_col in df.columns:
        return df[topic_label_col].astype(str)

    if topic_summary_col and topic_summary_col in df.columns:
        # If summary missing for a topic, fall back to topic id
        def _fmt(row):
            t = row[topic_col]
            s = row[topic_summary_col]
            if pd.notnull(s) and str(s).strip() and str(t) != "-1":
                return f"{t}. {str(s).strip()}"
            return str(t)

        return df.apply(_fmt, axis=1).astype(str)

    return df[topic_col].astype(str)


def sort_decades(decades: list[str]) -> list[str]:
    """Sort decades like '1590s', '1600s', 'Unknown' with Unknown at end."""
    known = []
    unknown = []
    for d in decades:
        if str(d).lower() == "unknown":
            unknown.append(d)
        else:
            known.append(d)

    def _key(v: str) -> int:
        try:
            return int(str(v).replace("s", ""))
        except Exception:
            return 10**9

    known_sorted = sorted(known, key=_key)
    return known_sorted + unknown


def topic_num(label: str) -> int:
    """Sort helper for labels that start with 'N.'"""
    try:
        return int(str(label).split(".", 1)[0])
    except Exception:
        return 999999


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        help="Path to the merged master CSV containing x/y/topic/topic_summary etc.",
        default="/Users/grace/Desktop/new meta/final chunk-level master table/chunk_level_master_table.csv",
    )
    ap.add_argument(
        "--out_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to save HTML outputs (default: this script's folder)",
    )
    ap.add_argument(
        "--keep_outliers",
        action="store_true",
        help="If set, do NOT filter topic == -1 (default: filter out -1).",
    )
    ap.add_argument(
        "--hover_chars",
        type=int,
        default=300,
        help="(Unused) Previously controlled chunk_text truncation; hover now shows only metadata.",
    )
    args = ap.parse_args()

    in_csv = args.in_csv
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"🔎 Loading: {in_csv}")
    df = pd.read_csv(in_csv)

    # Normalize column names (common issue: trailing/leading spaces from CSV merges)
    df.columns = [str(c).strip() for c in df.columns]

    print(f"✅ Loaded df: {df.shape}")

    # Required columns
    x_col = pick_col(df, ["x", "X"])
    y_col = pick_col(df, ["y", "Y"])
    topic_col = pick_col(df, ["topic", "Topic"])
    if not x_col or not y_col or not topic_col:
        raise ValueError(
            f"Missing required columns. Found x={x_col}, y={y_col}, topic={topic_col}. "
            f"Columns available: {list(df.columns)}"
        )

    # Optional / auto-detected columns
    # Prefer original metadata columns if present
    title_col = pick_col(df, ["title.1", "Title_HK", "title", "Title"])
    author_col = pick_col(df, ["author.1", "Author_HK", "author", "Author"])
    genre_col = pick_col(df, ["genre_brit_filter", "Genre", "genre", "genre_brit"])
    cluster_col = pick_col(df, ["cluster", "Cluster"])
    date_col = pick_col(df, [
        "date_first_performance_brit_filter",
        "date_first_performance",
        "Date",
        "date",
        "year",
    ])
    chunk_text_col = pick_col(df, ["chunk_text", "text_chunk", "text", "Text"])

    topic_label_col = pick_col(df, ["topic_label", "topicLabel", "Topic_Label"])
    topic_summary_col = pick_col(df, ["topic_summary", "Topic_Summary", "summary"])
    top_words_col = pick_col(df, ["top_words", "top_words", "topic_keywords", "keywords", "top_words"])
    count_col = pick_col(df, ["count", "topic_count", "Count"])

    # Clean / types
    df = df.copy()
    df[topic_col] = pd.to_numeric(df[topic_col], errors="coerce").fillna(-1).astype(int)

    # Optionally filter topic outliers
    if args.keep_outliers:
        df_valid = df.copy()
        print("ℹ️ keep_outliers=True → keeping topic == -1 rows")
    else:
        df_valid = df[df[topic_col] != -1].copy()
        print(f"✅ Filtered topic!=-1: {df_valid.shape}")

    # Remove baseline clustering noise clusters (0–5) from the interactive maps.
    # These clusters were identified during earlier cluster inspection as peripheral
    # materials that overly stretch the map and compress the main corpus into a corner.
    if cluster_col and cluster_col in df_valid.columns:
        df_valid[cluster_col] = pd.to_numeric(df_valid[cluster_col], errors="coerce")
        before_noise_filter = len(df_valid)
        df_valid = df_valid[~df_valid[cluster_col].isin([0, 1, 2, 3, 4, 5])].copy()
        removed_noise = before_noise_filter - len(df_valid)
        print(f"✅ Removed baseline clusters 0–5 from plotting: {removed_noise} rows removed; remaining {df_valid.shape}")
    else:
        print("⚠ No cluster column found; baseline clusters 0–5 cannot be removed from the interactive maps.")

    # Ensure topic_label exists
    df_valid["topic_label"] = build_topic_label(df_valid, topic_col, topic_label_col, topic_summary_col)

    # Build decade column if possible
    if date_col:
        years = df_valid[date_col].apply(safe_int)
        df_valid["Date_Decade"] = years.apply(lambda y: f"{(y // 10) * 10}s" if y is not None else "Unknown")
    else:
        df_valid["Date_Decade"] = "Unknown"

    # Colors by topic_label
    _, topic_uniques = pd.factorize(df_valid["topic_label"])
    color_palette = px.colors.qualitative.Plotly
    num_colors = len(color_palette)
    color_map = {label: color_palette[i % num_colors] for i, label in enumerate(topic_uniques)}
    marker_colors = [color_map[label] for label in df_valid["topic_label"]]

    # Hover text
    def _truncate(s: str, n: int) -> str:
        s = "" if s is None else str(s)
        if n and len(s) > n:
            return s[:n] + "…"
        return s

    # Build hover fields safely
    title_series = (
        df_valid[title_col].fillna("").astype(str)
        if title_col else pd.Series([""] * len(df_valid), index=df_valid.index)
    )
    author_series = (
        df_valid[author_col].fillna("").astype(str)
        if author_col else pd.Series([""] * len(df_valid), index=df_valid.index)
    )
    genre_series = df_valid[genre_col].astype(str) if genre_col else pd.Series([""] * len(df_valid), index=df_valid.index)
    date_series = df_valid[date_col].astype(str) if date_col else pd.Series([""] * len(df_valid), index=df_valid.index)
    # cluster_series = df_valid[cluster_col].astype(str) if cluster_col else pd.Series([""] * len(df_valid), index=df_valid.index)
    top_words_series = df_valid[top_words_col].fillna("").astype(str) if top_words_col else pd.Series([""] * len(df_valid), index=df_valid.index)
    df_valid["_hover"] = (
        "Title: " + title_series
        + "<br>Author: " + author_series
        + "<br>Date: " + date_series
        + "<br>Genre: " + genre_series
        + "<br>Topic: " + df_valid["topic_label"].astype(str)
        + "<br>Top words: " + top_words_series
    )

    # =================== Interactive date-highlighted topic map (classic style) ===================
    decades = df_valid["Date_Decade"].dropna().unique().tolist()
    decades = sort_decades([str(d) for d in decades])

    date_buttons = []
    date_buttons.append(
        dict(
            label="All decades",
            method="restyle",
            args=[
                {
                    "marker.size": [[4] * len(df_valid)],
                    "marker.opacity": [[0.6] * len(df_valid)],
                    "marker.symbol": [["circle"] * len(df_valid)],
                    "marker.line.width": [[0] * len(df_valid)],
                    "marker.line.color": [["rgba(0,0,0,0)"] * len(df_valid)],
                    "marker.color": [marker_colors],
                },
                [0],
            ],
        )
    )

    for d in decades:
        mask = df_valid["Date_Decade"].astype(str) == str(d)
        marker_symbol = ["diamond" if is_sel else "circle" for is_sel in mask]
        marker_size = [7 if is_sel else 4 for is_sel in mask]
        marker_opacity = [0.85 if is_sel else 0.6 for is_sel in mask]
        marker_line_width = [2 if is_sel else 0 for is_sel in mask]
        marker_line_color = ["black" if is_sel else "rgba(0,0,0,0)" for is_sel in mask]
        date_buttons.append(
            dict(
                label=str(d),
                method="restyle",
                args=[
                    {
                        "marker.size": [marker_size],
                        "marker.opacity": [marker_opacity],
                        "marker.symbol": [marker_symbol],
                        "marker.line.width": [marker_line_width],
                        "marker.line.color": [marker_line_color],
                        "marker.color": [marker_colors],
                    },
                    [0],
                ],
            )
        )

    fig_date = go.Figure()
    fig_date.add_trace(
        go.Scattergl(
            x=df_valid[x_col],
            y=df_valid[y_col],
            mode="markers",
            marker=dict(
                size=4,
                color=marker_colors,
                opacity=0.6,
                symbol="circle",
                line=dict(width=0, color="rgba(0,0,0,0)"),
            ),
            text=df_valid["_hover"],
            hovertemplate="%{text}",
        )
    )

    fig_date.update_layout(
        updatemenus=[
            dict(
                buttons=date_buttons,
                direction="down",
                x=1.15,
                y=1,
                showactive=True,
                active=0,
            )
        ],
        title="Topic-Clustered Renaissance Drama Corpus: Date Highlight",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    fig_date.update_xaxes(visible=False)
    fig_date.update_yaxes(visible=False)

    out_date = os.path.join(out_dir, "interactive_topics_date_highlight.html")
    fig_date.write_html(out_date, include_plotlyjs="cdn")
    print(f"✅ Saved: {out_date}")

    # =================== Interactive genre-highlighted topic map (classic style) ===================
    genres = []
    if genre_col:
        g_series = df_valid[genre_col].fillna("").astype(str).str.strip()
        g_series = g_series.replace({"": "Unknown"})
        # Normalize case-only variants so values like "Comedy" and "comedy" collapse.
        # Keep a readable display form using title case.
        df_valid["_Genre"] = g_series.str.lower().str.title()
        genres = sorted(df_valid["_Genre"].unique().tolist())
    else:
        print("⚠ No genre column found; genre dropdown will only contain 'All genres'.")
        df_valid["_Genre"] = "Unknown"
        genres = ["Unknown"]

    genre_buttons = []
    genre_buttons.append(
        dict(
            label="All genres",
            method="restyle",
            args=[
                {
                    "marker.size": [[4] * len(df_valid)],
                    "marker.opacity": [[0.6] * len(df_valid)],
                    "marker.symbol": [["circle"] * len(df_valid)],
                    "marker.line.width": [[0] * len(df_valid)],
                    "marker.line.color": [["rgba(0,0,0,0)"] * len(df_valid)],
                    "marker.color": [marker_colors],
                },
                [0],
            ],
        )
    )

    for g in genres:
        mask = df_valid["_Genre"].astype(str) == str(g)
        marker_symbol = ["diamond" if is_sel else "circle" for is_sel in mask]
        marker_size = [7 if is_sel else 4 for is_sel in mask]
        marker_opacity = [0.85 if is_sel else 0.6 for is_sel in mask]
        marker_line_width = [2 if is_sel else 0 for is_sel in mask]
        marker_line_color = ["black" if is_sel else "rgba(0,0,0,0)" for is_sel in mask]
        genre_buttons.append(
            dict(
                label=str(g),
                method="restyle",
                args=[
                    {
                        "marker.size": [marker_size],
                        "marker.opacity": [marker_opacity],
                        "marker.symbol": [marker_symbol],
                        "marker.line.width": [marker_line_width],
                        "marker.line.color": [marker_line_color],
                        "marker.color": [marker_colors],
                    },
                    [0],
                ],
            )
        )

    fig_genre = go.Figure()
    fig_genre.add_trace(
        go.Scattergl(
            x=df_valid[x_col],
            y=df_valid[y_col],
            mode="markers",
            marker=dict(
                size=4,
                color=marker_colors,
                opacity=0.6,
                symbol="circle",
                line=dict(width=0, color="rgba(0,0,0,0)"),
            ),
            text=df_valid["_hover"],
            hovertemplate="%{text}",
        )
    )

    fig_genre.update_layout(
        updatemenus=[
            dict(
                buttons=genre_buttons,
                direction="down",
                x=1.15,
                y=1,
                showactive=True,
                active=0,
            )
        ],
        title="Topic-Clustered Renaissance Drama Corpus: Genre Highlight",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    fig_genre.update_xaxes(visible=False)
    fig_genre.update_yaxes(visible=False)

    out_genre = os.path.join(out_dir, "interactive_topics_genre_highlight.html")
    fig_genre.write_html(out_genre, include_plotlyjs="cdn")
    print(f"✅ Saved: {out_genre}")

    # =================== Interactive author-highlighted topic map (classic style) ===================
    authors = []
    if author_col:
        # Treat empty strings as Unknown
        a_series = df_valid[author_col].fillna("").astype(str).str.strip()
        a_series = a_series.replace({"": "Unknown"})
        df_valid["_Author"] = a_series
        authors = sorted(df_valid["_Author"].unique().tolist())
    else:
        print("⚠ No author column found; author dropdown will only contain 'All authors'.")
        df_valid["_Author"] = "Unknown"
        authors = ["Unknown"]

    author_buttons = []
    author_buttons.append(
        dict(
            label="All authors",
            method="restyle",
            args=[
                {
                    "marker.size": [[4] * len(df_valid)],
                    "marker.opacity": [[0.6] * len(df_valid)],
                    "marker.symbol": [["circle"] * len(df_valid)],
                    "marker.line.width": [[0] * len(df_valid)],
                    "marker.line.color": [["rgba(0,0,0,0)"] * len(df_valid)],
                    "marker.color": [marker_colors],
                },
                [0],
            ],
        )
    )

    for a in authors:
        mask = df_valid["_Author"].astype(str) == str(a)
        marker_symbol = ["diamond" if is_sel else "circle" for is_sel in mask]
        marker_size = [7 if is_sel else 4 for is_sel in mask]
        marker_opacity = [0.85 if is_sel else 0.6 for is_sel in mask]
        marker_line_width = [2 if is_sel else 0 for is_sel in mask]
        marker_line_color = ["black" if is_sel else "rgba(0,0,0,0)" for is_sel in mask]
        author_buttons.append(
            dict(
                label=str(a),
                method="restyle",
                args=[
                    {
                        "marker.size": [marker_size],
                        "marker.opacity": [marker_opacity],
                        "marker.symbol": [marker_symbol],
                        "marker.line.width": [marker_line_width],
                        "marker.line.color": [marker_line_color],
                        "marker.color": [marker_colors],
                    },
                    [0],
                ],
            )
        )

    fig_author = go.Figure()
    fig_author.add_trace(
        go.Scattergl(
            x=df_valid[x_col],
            y=df_valid[y_col],
            mode="markers",
            marker=dict(
                size=4,
                color=marker_colors,
                opacity=0.6,
                symbol="circle",
                line=dict(width=0, color="rgba(0,0,0,0)"),
            ),
            text=df_valid["_hover"],
            hovertemplate="%{text}",
        )
    )

    fig_author.update_layout(
        updatemenus=[
            dict(
                buttons=author_buttons,
                direction="down",
                x=1.15,
                y=1,
                showactive=True,
                active=0,
            )
        ],
        title="Topic-Clustered Renaissance Drama Corpus: Author Highlight",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    fig_author.update_xaxes(visible=False)
    fig_author.update_yaxes(visible=False)

    out_author = os.path.join(out_dir, "interactive_topics_author_highlight.html")
    fig_author.write_html(out_author, include_plotlyjs="cdn")
    print(f"✅ Saved: {out_author}")

    # =================== Interactive title-highlighted topic map (classic style) ===================
    titles = []
    if title_col:
        t_series = df_valid[title_col].fillna("").astype(str).str.strip()
        t_series = t_series.replace({"": "Unknown"})
        df_valid["_Title"] = t_series
        titles = sorted(df_valid["_Title"].unique().tolist())
    else:
        print("⚠ No title column found; title dropdown will only contain 'All titles'.")
        df_valid["_Title"] = "Unknown"
        titles = ["Unknown"]

    title_buttons = []
    title_buttons.append(
        dict(
            label="All titles",
            method="restyle",
            args=[
                {
                    "marker.size": [[4] * len(df_valid)],
                    "marker.opacity": [[0.6] * len(df_valid)],
                    "marker.symbol": [["circle"] * len(df_valid)],
                    "marker.line.width": [[0] * len(df_valid)],
                    "marker.line.color": [["rgba(0,0,0,0)"] * len(df_valid)],
                    "marker.color": [marker_colors],
                },
                [0],
            ],
        )
    )

    for t in titles:
        mask = df_valid["_Title"].astype(str) == str(t)
        marker_symbol = ["diamond" if is_sel else "circle" for is_sel in mask]
        marker_size = [7 if is_sel else 4 for is_sel in mask]
        marker_opacity = [0.85 if is_sel else 0.6 for is_sel in mask]
        marker_line_width = [2 if is_sel else 0 for is_sel in mask]
        marker_line_color = ["black" if is_sel else "rgba(0,0,0,0)" for is_sel in mask]
        title_buttons.append(
            dict(
                label=str(t),
                method="restyle",
                args=[
                    {
                        "marker.size": [marker_size],
                        "marker.opacity": [marker_opacity],
                        "marker.symbol": [marker_symbol],
                        "marker.line.width": [marker_line_width],
                        "marker.line.color": [marker_line_color],
                        "marker.color": [marker_colors],
                    },
                    [0],
                ],
            )
        )

    fig_title = go.Figure()
    fig_title.add_trace(
        go.Scattergl(
            x=df_valid[x_col],
            y=df_valid[y_col],
            mode="markers",
            marker=dict(
                size=4,
                color=marker_colors,
                opacity=0.6,
                symbol="circle",
                line=dict(width=0, color="rgba(0,0,0,0)"),
            ),
            text=df_valid["_hover"],
            hovertemplate="%{text}",
        )
    )

    fig_title.update_layout(
        updatemenus=[
            dict(
                buttons=title_buttons,
                direction="down",
                x=1.15,
                y=1,
                showactive=True,
                active=0,
            )
        ],
        title="Topic-Clustered Renaissance Drama Corpus: Title Highlight",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    fig_title.update_xaxes(visible=False)
    fig_title.update_yaxes(visible=False)

    out_title = os.path.join(out_dir, "interactive_topics_title_highlight.html")
    fig_title.write_html(out_title, include_plotlyjs="cdn")
    print(f"✅ Saved: {out_title}")

    # =================== Interactive topic-highlighted topic map (classic style) ===================
    # Build an ordered list of topic labels (e.g., "16. ...") so the dropdown is sorted by topic id.
    topic_labels_list = sorted(df_valid["topic_label"].astype(str).unique().tolist(), key=topic_num)

    topic_buttons = []
    topic_buttons.append(
        dict(
            label="All topics",
            method="restyle",
            args=[
                {
                    "marker.size": [[4] * len(df_valid)],
                    "marker.opacity": [[0.6] * len(df_valid)],
                    "marker.symbol": [["circle"] * len(df_valid)],
                    "marker.line.width": [[0] * len(df_valid)],
                    "marker.line.color": [["rgba(0,0,0,0)"] * len(df_valid)],
                    "marker.color": [marker_colors],
                },
                [0],
            ],
        )
    )

    for tl in topic_labels_list:
        mask = df_valid["topic_label"].astype(str) == str(tl)
        marker_symbol = ["diamond" if is_sel else "circle" for is_sel in mask]
        marker_size = [7 if is_sel else 4 for is_sel in mask]
        marker_opacity = [0.85 if is_sel else 0.6 for is_sel in mask]
        marker_line_width = [2 if is_sel else 0 for is_sel in mask]
        marker_line_color = ["black" if is_sel else "rgba(0,0,0,0)" for is_sel in mask]
        topic_buttons.append(
            dict(
                label=str(tl),
                method="restyle",
                args=[
                    {
                        "marker.size": [marker_size],
                        "marker.opacity": [marker_opacity],
                        "marker.symbol": [marker_symbol],
                        "marker.line.width": [marker_line_width],
                        "marker.line.color": [marker_line_color],
                        "marker.color": [marker_colors],
                    },
                    [0],
                ],
            )
        )

    fig_topic = go.Figure()
    fig_topic.add_trace(
        go.Scattergl(
            x=df_valid[x_col],
            y=df_valid[y_col],
            mode="markers",
            marker=dict(
                size=4,
                color=marker_colors,
                opacity=0.6,
                symbol="circle",
                line=dict(width=0, color="rgba(0,0,0,0)"),
            ),
            text=df_valid["_hover"],
            hovertemplate="%{text}",
        )
    )

    fig_topic.update_layout(
        updatemenus=[
            dict(
                buttons=topic_buttons,
                direction="down",
                x=1.15,
                y=1,
                showactive=True,
                active=0,
            )
        ],
        title="Topic-Clustered Renaissance Drama Corpus: Topic Highlight",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    fig_topic.update_xaxes(visible=False)
    fig_topic.update_yaxes(visible=False)

    out_topic = os.path.join(out_dir, "interactive_topics_topic_highlight.html")
    fig_topic.write_html(out_topic, include_plotlyjs="cdn")
    print(f"✅ Saved: {out_topic}")

    print("🎉 Done.")


if __name__ == "__main__":
    main()