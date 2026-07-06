"""
15_divergence_analysis/code/per_author_divergence.py

Per-author thematic divergence from the rest of the genre.

For every genre (Comedy, Tragedy, History) and every author with at least
MIN_WORKS distinct plays (TCP) in that genre, this script measures how far the
author's topic distribution diverges from the pooled topic distribution of all
*other* authors in the same genre, using Jensen-Shannon Divergence (JSD, bits).

The result is rendered as a three-panel horizontal bar chart, with Shakespeare
highlighted, mirroring the per-genre layout used elsewhere in the project.

Inputs (relative to repo root):
  09_master_table_construction/results/chunk_level_master_table.csv

Outputs:
  15_divergence_analysis/results/per_author_divergence.csv
  15_divergence_analysis/results/per_author_divergence.png
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

MASTER  = f"{ROOT}/09_master_table_construction/results/chunk_level_master_table.csv"
OUT_DIR = f"{ROOT}/15_divergence_analysis/results"

# ── parameters ─────────────────────────────────────────────────────────────
GENRES        = ["Comedy", "Tragedy", "History"]
MIN_WORKS     = 3            # keep authors with at least this many plays in a genre
SHAKESPEARE   = "Shakespeare"
COLOR_SHA     = "#C0392B"    # red highlight for Shakespeare
COLOR_OTHER   = "#5B8CB7"    # blue for all other authors

# ── priority-based genre normalisation (mirrors csv_maker.py / compute_divergence.py) ──
# Handles compound labels like "tragedy;history" → "Tragedy", so plays with
# compound genre labels are not silently dropped.
_PRIORITY = ["tragedy", "tragicomedy", "comedy", "history"]

def _split_genre_tokens(raw):
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "null", "none listed", "not in britdrama"}:
        return []
    s = re.sub(r"[,/]+", ";", s)
    parts = [p.strip().lower() for p in s.split(";")]
    parts = [p for p in parts if p]
    normed = []
    for p in parts:
        p = p.replace("tragic-comedy", "tragicomedy")
        p = p.replace("tragic comedy", "tragicomedy")
        normed.append(p)
    return normed

def genre_main_priority(raw):
    tokens = _split_genre_tokens(raw)
    if not tokens:
        return "Other"
    for key in _PRIORITY:
        for t in tokens:
            if key in t:
                return key.capitalize()
    return "Other"

# ── load ───────────────────────────────────────────────────────────────────
master = pd.read_csv(MASTER)
master["genre_clean"] = master["genre_brit_filter"].apply(genre_main_priority)
master = master[master["genre_clean"].isin(GENRES) & master["author.1"].notna()].copy()


def surname(author):
    """Display label: the family name before the first comma."""
    return str(author).split(",", 1)[0].strip()


def is_named_author(author):
    """
    Keep only genuine, attributable authors. Drops anonymous credits and
    bracketed placeholder ids (e.g. "[84]") that are not real author names.
    """
    a = str(author).strip()
    return ("Anonymous" not in a) and (not a.startswith("["))


# ── core divergence functions (shared definition with compute_divergence.py) ─
def kl_div(p, q):
    """KL(p||q) in bits — p and q must already be smoothed and normalised."""
    return float(np.sum(p * np.log2(p / q)))


def jsd(p, q):
    """Jensen-Shannon divergence in bits (range 0-1)."""
    eps = 1e-10
    p = p + eps;  p /= p.sum()
    q = q + eps;  q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def topic_vector(df, topics):
    """Topic-count vector for a set of chunks over a fixed topic ordering."""
    counts = df["topic"].value_counts()
    return np.array([float(counts.get(t, 0.0)) for t in topics])


# ── compute per-author JSD vs the rest of the genre ─────────────────────────
records = []

for genre in GENRES:
    pool = master[master["genre_clean"] == genre]
    if pool.empty:
        continue

    # fixed topic ordering shared by author and "rest of genre" vectors
    topics = sorted(pool["topic"].unique())

    # number of works = distinct play titles, so duplicate editions of the
    # same play (different TCP ids) are not double-counted
    works_per_author = pool.groupby("author.1")["title.1"].nunique()
    keep_authors = [
        a for a, n in works_per_author.items()
        if n >= MIN_WORKS and is_named_author(a)
    ]

    for author in keep_authors:
        author_chunks = pool[pool["author.1"] == author]
        rest_chunks   = pool[pool["author.1"] != author]
        if rest_chunks.empty:
            continue

        p = topic_vector(author_chunks, topics)
        q = topic_vector(rest_chunks, topics)
        if p.sum() == 0 or q.sum() == 0:
            continue

        records.append({
            "genre": genre,
            "author": author,
            "label": surname(author),
            "n_works": int(works_per_author[author]),
            "n_chunks": int(len(author_chunks)),
            "jsd_bits": round(jsd(p, q), 4),
        })

result = pd.DataFrame(records).sort_values(
    ["genre", "jsd_bits"], ascending=[True, False]
).reset_index(drop=True)

os.makedirs(OUT_DIR, exist_ok=True)
result.to_csv(f"{OUT_DIR}/per_author_divergence.csv", index=False)

# ── plot: one horizontal-bar panel per genre ────────────────────────────────
fig, axes = plt.subplots(1, len(GENRES), figsize=(15, 5))
if len(GENRES) == 1:
    axes = [axes]

for ax, genre in zip(axes, GENRES):
    sub = result[result["genre"] == genre].sort_values("jsd_bits", ascending=True)
    if sub.empty:
        ax.set_visible(False)
        continue

    colors = [
        COLOR_SHA if SHAKESPEARE in lab else COLOR_OTHER
        for lab in sub["label"]
    ]
    y = np.arange(len(sub))
    ax.barh(y, sub["jsd_bits"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"])
    ax.set_title(genre, fontweight="bold")
    ax.set_xlabel("JSD vs rest of genre")

    # value annotation at the end of each bar
    for yi, val in zip(y, sub["jsd_bits"]):
        ax.text(val + 0.005, yi, f"{val:.2f}", va="center", fontsize=8)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

# shared legend
handles = [
    plt.Rectangle((0, 0), 1, 1, color=COLOR_SHA),
    plt.Rectangle((0, 0), 1, 1, color=COLOR_OTHER),
]
fig.legend(handles, ["Shakespeare", "Other authors"],
           loc="lower center", ncol=2, frameon=False)

fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(f"{OUT_DIR}/per_author_divergence.png", dpi=200, bbox_inches="tight")

print("Saved:")
print(f"  {OUT_DIR}/per_author_divergence.csv")
print(f"  {OUT_DIR}/per_author_divergence.png")
print()
for genre in GENRES:
    print(f"== {genre} ==")
    print(result[result["genre"] == genre][["label", "n_works", "jsd_bits"]]
          .to_string(index=False))
    print()
