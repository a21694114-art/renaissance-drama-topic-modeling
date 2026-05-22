import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import to_hex
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.offsetbox import VPacker, HPacker, TextArea, DrawingArea, AnnotationBbox
from matplotlib.patches import Rectangle

sns.set_theme(style="whitegrid", palette="muted")

# =========================
# Path configuration
# =========================

pivot_path = "/Users/grace/Desktop/new meta/genre/genre_topic_pivot_percent.csv"
summary_path = "/Users/grace/Desktop/new meta/gpt_topic/topic_summaries.csv"
output_dir = "/Users/grace/Desktop/new meta/genre visualization"

os.makedirs(output_dir, exist_ok=True)

# =========================
# Load CSV files
# =========================

df = pd.read_csv(pivot_path)
summary_df = pd.read_csv(summary_path)


# For visualization we focus on the three major genres; Tragicomedy remains in the data table but is not plotted
selected_genres = ["Comedy", "Tragedy", "History"]

# Keep only the selected genres
df = df[df["Genre_Main"].isin(selected_genres)]

# Remove the NumChunks column (already stored as float values)
df = df.drop(columns=["NumChunks"])

# Set Genre as the index
df = df.set_index("Genre_Main")

# Reorder genres according to the analysis order used in the paper
existing_genres = [g for g in selected_genres if g in df.index]
df = df.reindex(existing_genres)
selected_genres = existing_genres

# Normalize column names as strings
df.columns = df.columns.map(lambda x: str(x).strip())

# Extract all topic columns
topic_cols = [c for c in df.columns if c.isdigit()]
topic_cols_sorted = sorted(topic_cols, key=lambda x: int(x))

df = df[topic_cols_sorted]

# =========================
# Build Top-20 topic dictionary
# =========================

top_topics_per_genre = {}
for genre in selected_genres:
    genre_values = df.loc[genre]
    top20 = genre_values.sort_values(ascending=False).head(20)
    top_topics_per_genre[genre] = top20

# =========================
# Color mapping
# =========================

def reorder_paired_palette(palette):
    """Reorder paired palettes so visually similar light/dark neighbors are separated."""
    return palette[::2] + palette[1::2]

palette_full = []
palette_full += list(sns.color_palette("Dark2", 8))
palette_full += list(sns.color_palette("Set1", 9))
palette_full += list(sns.color_palette("Accent", 8))
palette_full += reorder_paired_palette(list(sns.color_palette("Paired", 12)))
palette_full += reorder_paired_palette(list(sns.color_palette("tab20", 20)))
palette_full += reorder_paired_palette(list(sns.color_palette("tab20b", 20)))
palette_full += reorder_paired_palette(list(sns.color_palette("tab20c", 20)))

# If the number of topics exceeds the qualitative palette size, extend with evenly spaced HSV colors.
if len(topic_cols_sorted) > len(palette_full):
    extra_needed = len(topic_cols_sorted) - len(palette_full)
    extra_colors = [cm.hsv(i / extra_needed) for i in range(extra_needed)]
    palette_full += extra_colors

palette_full = palette_full[:len(topic_cols_sorted)]
color_map = {t: palette_full[i] for i, t in enumerate(topic_cols_sorted)}

# Export the full topic color mapping
color_rows = []
for t in topic_cols_sorted:
    hexc = to_hex(color_map[t])
    color_rows.append((t, hexc))

pd.DataFrame(color_rows, columns=["Topic", "Color"]).to_csv(
    os.path.join(output_dir, "topic_color_map.csv"), index=False
)

# =========================
# (1) Stacked bar chart
# =========================

fig, ax = plt.subplots(figsize=(16, 6))

# Use slightly narrower bars to leave more room for the topic list
bar_width = 0.24
bottoms = np.zeros(len(selected_genres))
used_topics_order = []

for i, genre in enumerate(selected_genres):
    # Keep the bars compact while preserving space for the topic list
    x_pos = i * 0.34
    genre_values = df.loc[genre]
    top20 = genre_values.sort_values(ascending=False).head(20)
    sum_top20 = top20.sum()
    others = 100 - sum_top20

    cumulative = 0
    ordered_topics = top20.index.tolist()

    for topic in ordered_topics:
        value = top20[topic]
        ax.bar(x_pos, value, bar_width,
               bottom=bottoms[i] + cumulative,
               color=color_map[str(topic)],
               edgecolor='white',
               linewidth=0.5)

        ax.text(x_pos,
                bottoms[i] + cumulative + value/2,
                f"T{topic}",
                ha='center', va='center',
                fontsize=7, fontweight='bold',
                color='white')

        cumulative += value

        if topic not in used_topics_order:
            used_topics_order.append(topic)

    ax.bar(x_pos, others, bar_width,
           bottom=bottoms[i] + cumulative,
           color="#B0B0B0",
           edgecolor='white',
           linewidth=0.5)

    if "Others" not in used_topics_order:
        used_topics_order.append("Others")

ax.set_xticks([i * 0.34 for i in range(len(selected_genres))])
ax.set_xticklabels(selected_genres, fontsize=12)
ax.set_ylim(0, 100)
ax.set_ylabel('Percentage')
ax.set_xlabel('Genre')

# =========================
# Topic list on the right side (color block + T# + summary)
# =========================
# summary_df columns expected: topic, topic_summary
summary_lookup = dict(zip(summary_df['topic'].astype(str), summary_df['topic_summary'].astype(str)))

# used_topics_order stores topics that actually appear in the stacked bars
sorted_topics_for_legend = sorted(
    [t for t in used_topics_order if t != "Others"],
    key=lambda x: int(str(x))
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
    # Prevent long summaries from overflowing the figure (single-line display)
    max_chars = 95
    if len(summary) > max_chars:
        summary = summary[:max_chars - 1] + "…"

    # Colored square
    color_box = DrawingArea(10, 10, 0, 0)
    color_box.add_artist(Rectangle((0, 0), 10, 10, color=color))

    # Label text (single line without wrapping)
    text = TextArea(f" {label}: {summary}", textprops=dict(size=10, family="Arial", va="center"))

    row = HPacker(children=[color_box, text], align="center", pad=0, sep=2)
    summary_boxes.append(row)

# Split the topic list into two columns for readability
if summary_boxes:
    num_columns = 2
    chunks_per_col = int(np.ceil(len(summary_boxes) / num_columns))
    columns = [
        VPacker(children=summary_boxes[i * chunks_per_col: (i + 1) * chunks_per_col], align="left", pad=0, sep=2)
        for i in range(num_columns)
    ]
    legend_box = HPacker(children=columns, align="top", pad=5, sep=10)

    # Place the legend in the reserved right-side figure margin
    ab = AnnotationBbox(
        legend_box,
        (0.41, 0.92),
        xycoords=fig.transFigure,
        frameon=False,
        box_alignment=(0, 1)
    )
    ax.add_artist(ab)


# Make the bar chart narrower and reserve more space for the topic list
fig.subplots_adjust(left=0.06, right=0.36, top=0.92, bottom=0.16)

# Save two versions
plt.savefig(os.path.join(output_dir, "stacked_bar_top20.png"), dpi=300)
plt.savefig(os.path.join(output_dir, "stacked_bar_top20_notitle.png"), dpi=300)
plt.close()

# =========================
# (2) Horizontal bar chart
# =========================

summary_dict = dict(zip(summary_df['topic'].astype(str), summary_df['topic_summary']))

for genre, topics in top_topics_per_genre.items():
    plt.figure(figsize=(9, 5), dpi=300)

    topics = topics.copy()
    topics.index = topics.index.map(lambda x: str(x).strip())
    topics = topics.sort_values(ascending=True)
    sorted_topics = topics.index.tolist()

    topics.plot(kind='barh',
                color=[color_map[str(t)] for t in sorted_topics],
                edgecolor="white",
                linewidth=0.5)

    max_val = float(topics.max()) if len(topics) else 0
    x_upper = max(30, np.ceil(max_val + 18))
    plt.xlim(0, x_upper)
    plt.xticks(np.arange(0, x_upper + 1, 5))

    for i, (val, label) in enumerate(zip(topics.values, topics.index)):
        summary_text = summary_dict.get(str(label), f"Topic {label}")
        plt.text(val + 1, i - 0.1,
                 summary_text,
                 va='bottom', ha='left', fontsize=7)

    plt.xlabel('Percentage')
    plt.ylabel('Topic')

    plt.tight_layout()
    plt.subplots_adjust(right=0.82)

    plt.savefig(os.path.join(output_dir, f"{genre}_top20_barh.png"))
    plt.close()

print("🎉 All genre visualizations saved.")