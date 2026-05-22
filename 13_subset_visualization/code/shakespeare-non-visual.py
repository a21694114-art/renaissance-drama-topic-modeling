import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import to_hex
from matplotlib import cm
from matplotlib.offsetbox import VPacker, HPacker, TextArea, DrawingArea, AnnotationBbox
from matplotlib.patches import Rectangle

sns.set_theme(style="whitegrid", palette="muted")

# =========================
# 路径设置
# =========================
PIVOT_SHAKESPEARE = "/Users/grace/Desktop/new meta/non shakespeare_shakespeare/Shakespeare/genre_topic_pivot_percent.csv"
PIVOT_NON = "/Users/grace/Desktop/new meta/non shakespeare_shakespeare/Non-Shakespeare/genre_topic_pivot_percent.csv"
SUMMARY_PATH = "/Users/grace/Desktop/new meta/gpt_topic/topic_summaries.csv"
OUTPUT_ROOT = "/Users/grace/Desktop/new meta/non shakespeare_shakespeare visual"

# Reuse the canonical topic-color map generated in the main genre-visualization step
CANONICAL_COLOR_MAP = "/Users/grace/Desktop/new meta/genre visualization/topic_color_map.csv"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Main comparison in the paper focuses on the three analyzed genres only
SELECTED_GENRES = ["Comedy", "Tragedy", "History"]
TOP_N = 20


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



def build_fallback_colors(topic_cols_sorted):
    """Build additional colors only for topics missing from the canonical map."""
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

    if len(topic_cols_sorted) > len(palette_full):
        extra_needed = len(topic_cols_sorted) - len(palette_full)
        extra_colors = [cm.hsv(i / extra_needed) for i in range(extra_needed)]
        palette_full += extra_colors

    palette_full = palette_full[:len(topic_cols_sorted)]
    return {str(t): to_hex(palette_full[i]) for i, t in enumerate(topic_cols_sorted)}


def build_visuals(pivot_path: str, summary_path: str, output_dir: str, group_name: str):
    """Reproduce the same visualization logic as your old script for one subset."""

    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # 读取 CSV
    # =========================
    df = pd.read_csv(pivot_path)
    summary_df = pd.read_csv(summary_path)

    # 只保留需要的 Genre
    if "Genre_Main" not in df.columns:
        raise ValueError(f"Missing 'Genre_Main' in pivot: {pivot_path}")

    df = df[df["Genre_Main"].isin(SELECTED_GENRES)]

    # 防止 subset 缺某类导致报错：只画存在的 genres（一般不会缺）
    available_genres = [g for g in SELECTED_GENRES if g in set(df["Genre_Main"])]
    if not available_genres:
        raise ValueError(
            f"No selected genres found for {group_name}. "
            f"Found genres: {sorted(df['Genre_Main'].dropna().unique().tolist())}"
        )

    # 去掉 NumChunks 列
    if "NumChunks" in df.columns:
        df = df.drop(columns=["NumChunks"])

    # 设置 Genre 为索引
    df = df.set_index("Genre_Main")

    # Fix genre order for plotting and skip any missing rows safely
    available_genres = [g for g in SELECTED_GENRES if g in df.index]
    df = df.reindex(available_genres)

    # 列名统一为字符串
    df.columns = df.columns.map(lambda x: str(x).strip())

    # 获取所有 topic 列（digit）
    topic_cols = [c for c in df.columns if c.isdigit()]
    topic_cols_sorted = sorted(topic_cols, key=lambda x: int(x))
    df = df[topic_cols_sorted]

    # =========================
    # 颜色映射（优先复用主图的固定映射）
    # =========================
    canonical_map = load_canonical_color_map(CANONICAL_COLOR_MAP)
    fallback_map = build_fallback_colors(topic_cols_sorted)

    color_map = {}
    for t in topic_cols_sorted:
        t_str = str(t).strip()
        color_map[t_str] = canonical_map.get(t_str, fallback_map[t_str])

    # 导出本次实际使用的颜色映射
    color_rows = []
    for t in topic_cols_sorted:
        t_str = str(t).strip()
        hexc = str(color_map[t_str]).strip()
        color_rows.append((t_str, hexc))

    pd.DataFrame(color_rows, columns=["Topic", "Color"]).to_csv(
        os.path.join(output_dir, "topic_color_map.csv"), index=False
    )

    # =========================
    # 构建 Top20 字典
    # =========================
    top_topics_per_genre = {}
    for genre in available_genres:
        genre_values = df.loc[genre]
        top20 = genre_values.sort_values(ascending=False).head(TOP_N)
        top_topics_per_genre[genre] = top20

    # =========================
    # (1) 堆叠条形图
    # =========================
    fig, ax = plt.subplots(figsize=(16, 6))

    bar_width = 0.24
    bottoms = np.zeros(len(available_genres))
    used_topics_order = []

    for i, genre in enumerate(available_genres):
        x_pos = i * 0.34
        genre_values = df.loc[genre]
        top20 = genre_values.sort_values(ascending=False).head(TOP_N)
        sum_top20 = top20.sum()
        others = 100 - sum_top20

        cumulative = 0
        ordered_topics = top20.index.tolist()

        for topic in ordered_topics:
            value = top20[topic]
            ax.bar(
                x_pos,
                value,
                bar_width,
                bottom=bottoms[i] + cumulative,
                color=color_map[str(topic)],
                edgecolor="white",
                linewidth=0.5,
            )

            ax.text(
                x_pos,
                bottoms[i] + cumulative + value / 2,
                f"T{topic}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
            )

            cumulative += value

            if topic not in used_topics_order:
                used_topics_order.append(topic)

        ax.bar(
            x_pos,
            others,
            bar_width,
            bottom=bottoms[i] + cumulative,
            color="#B0B0B0",
            edgecolor="white",
            linewidth=0.5,
        )

        if "Others" not in used_topics_order:
            used_topics_order.append("Others")

    ax.set_xticks([i * 0.34 for i in range(len(available_genres))])
    ax.set_xticklabels(available_genres, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Genre")
    ax.set_title(
        f"Top 20 Topics per Selected Genre (Stacked Bar) — {group_name}",
        fontsize=18,
        fontweight="bold",
    )

    # =========================
    # 右侧主题列表（色块 + T# + summary）
    # =========================
    summary_lookup = dict(zip(summary_df["topic"].astype(str), summary_df["topic_summary"].astype(str)))

    sorted_topics_for_legend = sorted(
        [t for t in used_topics_order if t != "Others"],
        key=lambda x: int(str(x)),
    )
    if "Others" in used_topics_order:
        sorted_topics_for_legend.append("Others")

    summary_boxes = []
    for topic in sorted_topics_for_legend:
        if topic == "Others":
            continue

        t_str = str(topic).strip()
        color = to_hex(color_map[t_str])
        label = f"T{t_str}"
        summary = summary_lookup.get(t_str, "[summary missing]")

        max_chars = 95
        if len(summary) > max_chars:
            summary = summary[: max_chars - 1] + "…"

        color_box = DrawingArea(10, 10, 0, 0)
        color_box.add_artist(Rectangle((0, 0), 10, 10, color=color))

        text = TextArea(
            f" {label}: {summary}",
            textprops=dict(size=10, family="Arial", va="center"),
        )

        row = HPacker(children=[color_box, text], align="center", pad=0, sep=2)
        summary_boxes.append(row)

    if summary_boxes:
        num_columns = 2
        chunks_per_col = int(np.ceil(len(summary_boxes) / num_columns))
        columns = [
            VPacker(
                children=summary_boxes[i * chunks_per_col: (i + 1) * chunks_per_col],
                align="left",
                pad=0,
                sep=2,
            )
            for i in range(num_columns)
        ]
        legend_box = HPacker(children=columns, align="top", pad=5, sep=10)

        ab = AnnotationBbox(
            legend_box,
            (0.41, 0.92),
            xycoords=fig.transFigure,
            frameon=False,
            box_alignment=(0, 1),
        )
        ax.add_artist(ab)

    fig.subplots_adjust(left=0.06, right=0.36, top=0.92, bottom=0.16)

    # 保存两个版本
    plt.savefig(os.path.join(output_dir, "stacked_bar_top20.png"), dpi=300)
    ax.set_title("")
    plt.savefig(os.path.join(output_dir, "stacked_bar_top20_notitle.png"), dpi=300)
    plt.close()

    # =========================
    # (2) 横向条形图
    # =========================
    summary_dict = dict(zip(summary_df["topic"].astype(str), summary_df["topic_summary"]))

    for genre, topics in top_topics_per_genre.items():
        plt.figure(figsize=(8, 5), dpi=300)

        topics = topics.copy()
        topics.index = topics.index.map(lambda x: str(x).strip())
        topics = topics.sort_values(ascending=True)
        sorted_topics = topics.index.tolist()

        topics.plot(
            kind="barh",
            color=[color_map[str(t)] for t in sorted_topics],
            edgecolor="white",
            linewidth=0.5,
        )

        max_val = float(topics.max()) if len(topics) else 0
        x_upper = max(30, np.ceil(max_val + 18))
        plt.xlim(0, x_upper)
        plt.xticks(np.arange(0, x_upper + 1, 5))

        for i, (val, label) in enumerate(zip(topics.values, topics.index)):
            summary_text = summary_dict.get(str(label), f"Topic {label}")
            plt.text(
                val + 1,
                i - 0.1,
                summary_text,
                va="bottom",
                ha="left",
                fontsize=7,
            )

        plt.title(f"Top {TOP_N} Topics in {genre} — {group_name}", fontsize=16, fontweight="bold")
        plt.xlabel("Percentage")
        plt.ylabel("Topic")

        plt.tight_layout()
        plt.subplots_adjust(right=0.82)

        plt.savefig(os.path.join(output_dir, f"{genre}_top{TOP_N}_barh.png"))
        plt.close()

    print(f"🎉 Saved visualizations for {group_name}: {output_dir}")


def main():
    sh_out = os.path.join(OUTPUT_ROOT, "Shakespeare")
    non_out = os.path.join(OUTPUT_ROOT, "Non-Shakespeare")

    build_visuals(PIVOT_SHAKESPEARE, SUMMARY_PATH, sh_out, "Shakespeare")
    build_visuals(PIVOT_NON, SUMMARY_PATH, non_out, "Non-Shakespeare")

    print("🎉 All genre visualizations saved.")


if __name__ == "__main__":
    main()