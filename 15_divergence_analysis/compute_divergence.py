"""
15_divergence_analysis/code/compute_divergence.py

Computes Jensen-Shannon Divergence (JSD) and KL contributions between:
  (1) Genre pairs: Comedy vs Tragedy, Comedy vs History, Tragedy vs History
  (2) Shakespeare vs Non-Shakespeare within each genre

Also computes 95% bootstrap confidence intervals by resampling at the play
(TCP) level to account for sampling variance from unequal corpus sizes.

Inputs (relative to repo root):
  09_master_table_construction/results/chunk_level_master_table.csv
  11_genre_visualization/results/genre_topic_pivot_percent.csv
  11_genre_visualization/results/genre_topic_counts_long.csv
  12_comparative_analysis/results/Shakespeare/genre_topic_pivot_percent.csv
  12_comparative_analysis/results/Non-Shakespeare/genre_topic_pivot_percent.csv

Outputs:
  15_divergence_analysis/results/jsd_genre_pairs.csv
  15_divergence_analysis/results/jsd_shakespeare_vs_nonshakespeare.csv
  15_divergence_analysis/results/kl_top5_genre_pairs.csv
  15_divergence_analysis/results/kl_top5_shakespeare_vs_nonshakespeare.csv
"""

import re
import pandas as pd
import numpy as np
import os

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

MASTER    = f"{ROOT}/09_master_table_construction/results/chunk_level_master_table.csv"
LABELS    = f"{ROOT}/11_genre_visualization/results/genre_topic_counts_long.csv"
GENRE_PIV = f"{ROOT}/11_genre_visualization/results/genre_topic_pivot_percent.csv"
SHA_PIV   = f"{ROOT}/12_comparative_analysis/results/Shakespeare/genre_topic_pivot_percent.csv"
NSHA_PIV  = f"{ROOT}/12_comparative_analysis/results/Non-Shakespeare/genre_topic_pivot_percent.csv"
OUT_DIR   = f"{ROOT}/15_divergence_analysis/results"

# ── load ───────────────────────────────────────────────────────────────────
master    = pd.read_csv(MASTER)
labels_df = pd.read_csv(LABELS)
genre_piv = pd.read_csv(GENRE_PIV)
sha_piv   = pd.read_csv(SHA_PIV)
nsha_piv  = pd.read_csv(NSHA_PIV)

# topic id → label
topic_label_map = (
    labels_df[["topic", "topic_label"]]
    .drop_duplicates("topic")
    .set_index("topic")["topic_label"]
    .to_dict()
)
def short_label(tid):
    raw = topic_label_map.get(tid, str(tid))
    parts = str(raw).split(". ", 1)
    return parts[1] if len(parts) == 2 else raw

# ── priority-based genre normalisation (mirrors csv_maker.py) ──────────────
# Handles compound labels like "tragedy;history" → "Tragedy"
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

master["genre_clean"] = master["genre_brit_filter"].apply(genre_main_priority)
master["is_shakespeare"] = master["author.1"].str.contains("Shakespeare", na=False)

# union of all topic IDs across all pivot tables
topic_cols = sorted(
    set(c for c in genre_piv.columns if c not in ("Genre_Main", "NumChunks")) |
    set(c for c in sha_piv.columns   if c not in ("Genre_Main", "NumChunks")) |
    set(c for c in nsha_piv.columns  if c not in ("Genre_Main", "NumChunks")),
    key=lambda x: int(x)
)
all_topic_ints = [int(t) for t in topic_cols]
all_labels = [short_label(t) for t in all_topic_ints]

# ── core functions ─────────────────────────────────────────────────────────
def row_to_vec(df, genre_name):
    row = df[df["Genre_Main"] == genre_name]
    if row.empty:
        return np.zeros(len(topic_cols))
    row = row.iloc[0]
    return np.array([float(row.get(t, 0.0)) for t in topic_cols])

def kl_div(p, q):
    """KL(p||q) in bits — p and q must already be smoothed and normalised."""
    return float(np.sum(p * np.log2(p / q)))

def jsd(p, q):
    """Jensen-Shannon divergence in bits (range 0–1)."""
    eps = 1e-10
    p = p + eps;  p /= p.sum()
    q = q + eps;  q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

def jsd_and_topk(p, q, k=5):
    """Return (jsd_value, dataframe of top-k topics by JSD contribution)."""
    eps = 1e-10
    p = p + eps;  p /= p.sum()
    q = q + eps;  q /= q.sum()
    m = 0.5 * (p + q)
    contrib = 0.5 * p * np.log2(p / m) + 0.5 * q * np.log2(q / m)
    jsd_val = float(0.5 * kl_div(p, m) + 0.5 * kl_div(q, m))
    df = pd.DataFrame({
        "topic_id":    all_topic_ints,
        "topic_label": all_labels,
        "P_pct":       np.round(p * 100, 3),
        "Q_pct":       np.round(q * 100, 3),
        "jsd_contrib": np.round(contrib, 6),
    }).nlargest(k, "jsd_contrib").reset_index(drop=True)
    return jsd_val, df

def bootstrap_jsd(sub_p, sub_q, n_boot=1000, seed=42):
    """
    95% bootstrap CI for JSD, resampling at the play (TCP) level.
    sub_p / sub_q: filtered rows of master table for each group.
    """
    rng = np.random.default_rng(seed)
    plays_p = sub_p["TCP"].unique()
    plays_q = sub_q["TCP"].unique()
    boots = []
    for _ in range(n_boot):
        sp = rng.choice(plays_p, size=len(plays_p), replace=True)
        sq = rng.choice(plays_q, size=len(plays_q), replace=True)
        rp = sub_p[sub_p["TCP"].isin(sp)]
        rq = sub_q[sub_q["TCP"].isin(sq)]
        vp = np.array([(rp["topic"] == t).sum() for t in all_topic_ints], dtype=float)
        vq = np.array([(rq["topic"] == t).sum() for t in all_topic_ints], dtype=float)
        if vp.sum() == 0 or vq.sum() == 0:
            continue
        vp /= vp.sum();  vq /= vq.sum()
        boots.append(jsd(vp, vq))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

# ══════════════════════════════════════════════════════════════════════════
# 1. Genre-pair JSD
# ══════════════════════════════════════════════════════════════════════════
genre_pairs = [("Comedy", "Tragedy"), ("Comedy", "History"), ("Tragedy", "History")]

rows_jsd, rows_topk = [], []

for g1, g2 in genre_pairs:
    p = row_to_vec(genre_piv, g1)
    q = row_to_vec(genre_piv, g2)
    jsd_val, topk = jsd_and_topk(p, q)

    m1 = master[master["genre_clean"] == g1]
    m2 = master[master["genre_clean"] == g2]
    ci_lo, ci_hi = bootstrap_jsd(m1, m2)

    n1 = int(genre_piv[genre_piv["Genre_Main"] == g1]["NumChunks"].values[0])
    n2 = int(genre_piv[genre_piv["Genre_Main"] == g2]["NumChunks"].values[0])

    rows_jsd.append({
        "comparison": f"{g1} vs {g2}",
        "group_P": g1, "n_chunks_P": n1,
        "group_Q": g2, "n_chunks_Q": n2,
        "jsd_bits": round(jsd_val, 4),
        "ci_lo_95": round(ci_lo, 4),
        "ci_hi_95": round(ci_hi, 4),
    })
    for _, r in topk.iterrows():
        rows_topk.append({
            "comparison": f"{g1} vs {g2}",
            "rank": int(r.name) + 1,
            "topic_id": int(r.topic_id),
            "topic_label": r.topic_label,
            "P_pct": r.P_pct, "Q_pct": r.Q_pct,
            "jsd_contrib": r.jsd_contrib,
        })

df_jsd_genre = pd.DataFrame(rows_jsd)
df_topk_genre = pd.DataFrame(rows_topk)

# ══════════════════════════════════════════════════════════════════════════
# 2. Shakespeare vs Non-Shakespeare within each genre
# ══════════════════════════════════════════════════════════════════════════
rows_jsd2, rows_topk2 = [], []

for genre in ["Comedy", "Tragedy", "History"]:
    p = row_to_vec(sha_piv, genre)
    q = row_to_vec(nsha_piv, genre)
    jsd_val, topk = jsd_and_topk(p, q)

    m_sha  = master[(master["genre_clean"] == genre) &  master["is_shakespeare"]]
    m_nsha = master[(master["genre_clean"] == genre) & ~master["is_shakespeare"]]
    ci_lo, ci_hi = bootstrap_jsd(m_sha, m_nsha)

    n_sha  = int(sha_piv[sha_piv["Genre_Main"]  == genre]["NumChunks"].values[0])
    n_nsha = int(nsha_piv[nsha_piv["Genre_Main"] == genre]["NumChunks"].values[0])
    p_sha  = int(m_sha["TCP"].nunique())
    p_nsha = int(m_nsha["TCP"].nunique())

    rows_jsd2.append({
        "genre": genre,
        "n_plays_shakespeare": p_sha,   "n_chunks_shakespeare": n_sha,
        "n_plays_nonshakespeare": p_nsha, "n_chunks_nonshakespeare": n_nsha,
        "jsd_bits": round(jsd_val, 4),
        "ci_lo_95": round(ci_lo, 4),
        "ci_hi_95": round(ci_hi, 4),
    })
    for _, r in topk.iterrows():
        rows_topk2.append({
            "genre": genre,
            "rank": int(r.name) + 1,
            "topic_id": int(r.topic_id),
            "topic_label": r.topic_label,
            "shakespeare_pct": r.P_pct,
            "nonshakespeare_pct": r.Q_pct,
            "jsd_contrib": r.jsd_contrib,
        })

df_jsd_sha = pd.DataFrame(rows_jsd2)
df_topk_sha = pd.DataFrame(rows_topk2)

# ══════════════════════════════════════════════════════════════════════════
# 3. Save results
# ══════════════════════════════════════════════════════════════════════════
df_jsd_genre.to_csv(f"{OUT_DIR}/jsd_genre_pairs.csv", index=False)
df_topk_genre.to_csv(f"{OUT_DIR}/kl_top5_genre_pairs.csv", index=False)
df_jsd_sha.to_csv(f"{OUT_DIR}/jsd_shakespeare_vs_nonshakespeare.csv", index=False)
df_topk_sha.to_csv(f"{OUT_DIR}/kl_top5_shakespeare_vs_nonshakespeare.csv", index=False)

print("Results saved to 15_divergence_analysis/results/")
print("\njsd_genre_pairs.csv:")
print(df_jsd_genre.to_string(index=False))
print("\njsd_shakespeare_vs_nonshakespeare.csv:")
print(df_jsd_sha.to_string(index=False))
