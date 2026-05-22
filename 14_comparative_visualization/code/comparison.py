import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex
from matplotlib import cm
import numpy as np

sns.set_theme(style="whitegrid", palette="muted")

# ============================
# Paths (NEW DATA)
# ============================
PIVOT_SHAKESPEARE = "/Users/grace/Desktop/new meta/non shakespeare_shakespeare/Shakespeare/genre_topic_pivot_percent.csv"
PIVOT_NON = "/Users/grace/Desktop/new meta/non shakespeare_shakespeare/Non-Shakespeare/genre_topic_pivot_percent.csv"
SUMMARY_CSV = "/Users/grace/Desktop/new meta/gpt_topic/topic_summaries.csv"

# Output directory requested
OUT_DIR = "/Users/grace/Desktop/new meta/non shakespeare_shakespeare visual/comparison"

# Reuse the canonical topic-color map generated in the main genre-visualization step
CANONICAL_COLOR_MAP = "/Users/grace/Desktop/new meta/genre visualization/topic_color_map.csv"

# Main comparison focuses on the three analyzed genres only
SELECTED_GENRES = ["Comedy", "Tragedy", "History"]

# Dynamic x-limits are computed from the plotted values for each genre

TOP_N = 20


# ============================
# Helpers
# ============================

def load_summary_dict(summary_csv: str) -> dict:
    """topic -> topic_summary"""
    if not os.path.exists(summary_csv):
        return {}
    df = pd.read_csv(summary_csv)
    # expected columns: topic, topic_summary
    if "topic" not in df.columns or "topic_summary" not in df.columns:
        return {}
    return dict(zip(df["topic"].astype(str), df["topic_summary"].astype(str)))


def _read_pivot(pivot_path: str) -> pd.DataFrame:
    """Read pivot_percent CSV produced by csv_maker.py.

    It was saved with Genre_Main as index; first column may be unnamed.
    Returns a DataFrame indexed by Genre_Main.
    """
    df = pd.read_csv(pivot_path)

    # If the file has an unnamed first column (index), use it as Genre_Main
    if "Genre_Main" in df.columns:
        df = df.set_index("Genre_Main")
    else:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Genre_Main"}).set_index("Genre_Main")

    # Drop NumChunks if present
    if "NumChunks" in df.columns:
        df = df.drop(columns=["NumChunks"])

    # Ensure topic columns are strings
    df.columns = df.columns.map(lambda x: str(x).strip())

    # Keep only digit columns as topics
    topic_cols = [c for c in df.columns if c.isdigit()]
    topic_cols_sorted = sorted(topic_cols, key=lambda x: int(x))
    df = df[topic_cols_sorted]

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df


def load_genre_row(pivot_df: pd.DataFrame, genre: str) -> pd.Series:
    """Return a single genre row (topic percentages) from pivot_df."""
    if genre not in pivot_df.index:
        raise KeyError(f"Genre '{genre}' not found. Available: {list(pivot_df.index)}")

    row = pivot_df.loc[genre]
    row.index = row.index.map(lambda x: str(x).strip())
    return row


def topn(series: pd.Series, n: int = TOP_N) -> pd.Series:
    return series.sort_values(ascending=False).head(n)


def load_canonical_color_map(path: str) -> dict:
    """Load a previously exported topic->color mapping if available."""
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "Topic" not in df.columns or "Color" not in df.columns:
        return {}
    return {
        str(topic).strip(): str(color).strip()
        for topic, color in zip(df["Topic"], df["Color"])
        if pd.notna(topic) and pd.notna(color)
    }


def build_fallback_color_map(topics: list) -> dict:
    """Build fallback colors only for topics missing from the canonical map."""
    def reorder_paired_palette(palette):
        return palette[::2] + palette[1::2]

    palette_full = []
    palette_full += list(sns.color_palette("Dark2", 8))
    palette_full += list(sns.color_palette("Set1", 9))
    palette_full += list(sns.color_palette("Accent", 8))
    palette_full += reorder_paired_palette(list(sns.color_palette("Paired", 12)))
    palette_full += reorder_paired_palette(list(sns.color_palette("tab20", 20)))
    palette_full += reorder_paired_palette(list(sns.color_palette("tab20b", 20)))
    palette_full += reorder_paired_palette(list(sns.color_palette("tab20c", 20)))

    if len(topics) > len(palette_full):
        extra_needed = len(topics) - len(palette_full)
        extra_colors = [cm.hsv(i / extra_needed) for i in range(extra_needed)]
        palette_full += extra_colors

    palette_full = palette_full[:len(topics)]
    return {str(t): to_hex(palette_full[i]) for i, t in enumerate(topics)}


def plot_topn_barh(
    ax,
    top_series: pd.Series,
    color_map: dict,
    summary_dict: dict,
    xlim=(0, 75),
    panel_label=None,
    force_top_label_above: bool = False,
):
    """Publication-style horizontal bar chart with summaries; no topic IDs on y-axis."""

    s = top_series.copy().sort_values(ascending=True)
    topics = [str(t).strip() for t in s.index.tolist()]

    ax.barh(
        y=range(len(s)),
        width=s.values,
        color=[color_map[str(t)] for t in topics],
        edgecolor="white",
        linewidth=0.5,
    )

    # Remove topic IDs from the y-axis for publication-style figure
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([])

    ax.set_xlim(*xlim)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 5))
    ax.set_xlabel("Percentage")
    ax.set_ylabel("")
    ax.margins(y=0.02)

    max_val = float(s.max()) if len(s) else 0.0
    max_i = len(s) - 1  # because s is sorted ascending; the largest bar is the last row

    # Add summary text with overflow-safe placement
    x0, x1 = ax.get_xlim()
    right_pad = 2.0
    edge_pad = 3.0

    for i, (val, topic_id) in enumerate(zip(s.values, topics)):
        txt = summary_dict.get(str(topic_id), f"Topic {topic_id}")

        # Manual override: for the single top bar in a panel, place the label above the bar.
        if force_top_label_above and i == max_i:
            ax.text(
                x1 - 10,
                i + 0.40,
                txt,
                va="bottom",
                ha="right",
                fontsize=7,
                color="black",
                clip_on=True,
            )
            continue

        x = val + right_pad
        ha = "left"
        # Robust overflow detection:
        # 1) if the bar end is close to the right edge (even moderate-length text can spill), OR
        # 2) if the default right-of-bar placement would exceed the edge, OR
        # 3) if the label is long and we're already in the right half.
        near_edge = val > (x1 - 8)
        will_overflow = near_edge or (x > (x1 - edge_pad)) or (len(txt) > 70 and x > (x1 - 20))
        if will_overflow:
            # Place text ABOVE the bar, but anchor it to the right edge so it never spills out.
            ax.text(
                x1 - 10,
                i + 0.40,
                txt,
                va="bottom",
                ha="right",
                fontsize=7,
                color="black",
                clip_on=True,
            )
            continue

        ax.text(
            x,
            i + 0.15,
            txt,
            va="center",
            ha=ha,
            fontsize=7,
            color="black",
            clip_on=True,
        )

    # Panel label (no big title)
    if panel_label:
        ax.text(
            -0.04,
            1.01,
            panel_label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
        )


def make_one_genre_comparison(
    genre: str,
    pivot_non: pd.DataFrame,
    pivot_sh: pd.DataFrame,
    summary_dict: dict,
    out_dir: str,
):
    non_row = load_genre_row(pivot_non, genre)
    sh_row = load_genre_row(pivot_sh, genre)

    non_top = topn(non_row, TOP_N)
    sh_top = topn(sh_row, TOP_N)

    # shared colors cover both panels
    all_topics = sorted(
        set(non_top.index.astype(str)) | set(sh_top.index.astype(str)),
        key=lambda x: int(x),
    )
    canonical_map = load_canonical_color_map(CANONICAL_COLOR_MAP)
    fallback_map = build_fallback_color_map(all_topics)
    color_map = {
        str(t).strip(): canonical_map.get(str(t).strip(), fallback_map[str(t).strip()])
        for t in all_topics
    }

    # dynamic x-limit based on the larger maximum across both subsets
    max_val = max(float(non_top.max()) if len(non_top) else 0.0,
                  float(sh_top.max()) if len(sh_top) else 0.0)
    x_upper = max(30, np.ceil(max_val + 18))
    xlim = (0, x_upper)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300, sharex=True)

    # Left: Non-Shakespeare; Right: Shakespeare
    plot_topn_barh(
        axes[0],
        non_top,
        color_map,
        summary_dict,
        xlim=xlim,
        panel_label="(a) Non-Shakespeare",
    )
    plot_topn_barh(
        axes[1],
        sh_top,
        color_map,
        summary_dict,
        xlim=xlim,
        panel_label="(b) Shakespeare",
        force_top_label_above=(genre == "History"),
    )

    # No big title; keep layout tight
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"Fig_comparison_{genre}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("✅ Saved:", out_path)


# ============================
# Run: generate 4 comparison figures
# ============================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pivot_non = _read_pivot(PIVOT_NON)
    pivot_sh = _read_pivot(PIVOT_SHAKESPEARE)

    # keep only the main genres (safety)
    pivot_non = pivot_non.loc[[g for g in SELECTED_GENRES if g in pivot_non.index]]
    pivot_sh = pivot_sh.loc[[g for g in SELECTED_GENRES if g in pivot_sh.index]]

    summary_dict = load_summary_dict(SUMMARY_CSV)

    for genre in SELECTED_GENRES:
        # Skip if a genre is missing in either subset
        if genre not in pivot_non.index or genre not in pivot_sh.index:
            print(f"⚠️ Skipping {genre} (missing in one subset)")
            continue

        make_one_genre_comparison(
            genre=genre,
            pivot_non=pivot_non,
            pivot_sh=pivot_sh,
            summary_dict=summary_dict,
            out_dir=OUT_DIR,
        )

    print("\n🎨 All comparison figures generated successfully!")


if __name__ == "__main__":
    main()