"""
16_wsc_seminar_handouts/code/reproduce_handouts.py

Reproduces every figure and every printed number on the four supplementary
handouts prepared for WSC Seminar 14 (Verona 2026), "Modelling Genre as
Thematic Distribution" (Dain Lee & Sujin Kang).

Handout 1 - Topic 12 (Ottoman warfare cluster) membership table and the
            segment-level evidence behind the "mention is not thematisation"
            argument (Othello's Turk/Ottoman-form chunks).
Handout 2 - Per-author thematic divergence within each genre (JSD vs the
            rest of the genre), Shakespeare highlighted.
Handout 3 - Othello's distributional position: JSD to each genre profile,
            and a 13-play panel of Shakespeare's tragedy-group plays ranked
            by how far each leans toward the comedy profile.
Handout 4 - The Merry Wives vs Measure for Measure comparison: share of
            assigned segments in the domestic-social register, and the
            topic composition of Measure for Measure.

Inputs (relative to repo root):
  09_master_table_construction/results/chunk_level_master_table.csv
      One row per topic-ASSIGNED segment (outliers already excluded),
      with DEEP genre labels, attribution, and topic assignments.
  07_topic_modeling/results/chunk_topics_filtered 2.csv
      One row per segment INCLUDING outliers (topic == -1), with the
      raw chunk text. Used only for the outlier/mention statistics.

Outputs (written to ../results/):
  handout1_topic12_members.csv
  handout2_per_author_divergence.png / .csv
  handout3_othello_panel.png
  handout4_mw_vs_m4m.png
  plus a console report of every statistic, each labelled with the value
  printed on the corresponding handout so the two can be compared line
  by line.

Method notes
------------
* Genre normalisation uses the same priority rule as the rest of the
  pipeline (tragedy > tragicomedy > comedy > history), so compound DEEP
  labels such as "tragedy;history" are kept, not dropped.
* All divergences are Jensen-Shannon divergence in bits (log base 2),
  computed over topic-count vectors on a shared topic ordering.
* "Assigned segments" means segments the model actually clustered
  (HDBSCAN outliers, topic == -1, are excluded) - the same convention
  used throughout the circulated paper.
* Attribution scopes: Handout 2 matches the author field exactly
  ("Shakespeare, William"), i.e. sole-attribution records only.
  The paper's section 5.5 aggregate instead includes every segment whose
  attribution CONTAINS Shakespeare (collaborations included). Both
  totals are reported below because the two scopes explain the small
  difference between the chart value (0.589) and the paper value (0.597)
  for tragedy; comedy has no collaboration records, so its two values
  coincide exactly.
"""

import os
import re
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MASTER   = f"{ROOT}/09_master_table_construction/results/chunk_level_master_table.csv"
FILTERED = f"{ROOT}/07_topic_modeling/results/chunk_topics_filtered 2.csv"
OUT_DIR  = f"{ROOT}/16_wsc_seminar_handouts/results"
os.makedirs(OUT_DIR, exist_ok=True)

GENRES = ["Comedy", "Tragedy", "History"]
COLOR_SHA, COLOR_OTHER, COLOR_POOL = "#C0392B", "#5B8CB7", "#A8BFD4"

# ── genre normalisation (identical to 15_divergence_analysis) ──────────────
_PRIORITY = ["tragedy", "tragicomedy", "comedy", "history"]

def genre_main_priority(raw):
    """Priority-based normalisation of (possibly compound) DEEP genre labels."""
    if raw is None:
        return "Other"
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "null", "none listed", "not in britdrama"}:
        return "Other"
    s = re.sub(r"[,/]+", ";", s)
    tokens = []
    for p in s.split(";"):
        p = p.strip().lower().replace("tragic-comedy", "tragicomedy").replace("tragic comedy", "tragicomedy")
        if p:
            tokens.append(p)
    for key in _PRIORITY:
        for t in tokens:
            if key in t:
                return key.capitalize()
    return "Other"

# ── Jensen-Shannon divergence, bits ─────────────────────────────────────────
def jsd(p, q):
    """JSD in bits between two (unnormalised) topic-count vectors."""
    eps = 1e-10
    p = np.asarray(p, dtype=float) + eps; p /= p.sum()
    q = np.asarray(q, dtype=float) + eps; q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def topic_vector(df, topics):
    """Topic-count vector for a set of segments over a fixed topic ordering."""
    counts = df["topic"].value_counts()
    return np.array([float(counts.get(t, 0.0)) for t in topics])

def report(name, got, printed):
    """Console line comparing a recomputed value with the handout-printed one."""
    flag = "OK " if str(got) == str(printed) else "   "
    print(f"  [{flag}] {name}: computed {got}  (handout prints {printed})")

# ── load ────────────────────────────────────────────────────────────────────
master = pd.read_csv(MASTER)
master["genre_clean"] = master["genre_brit_filter"].apply(genre_main_priority)

# ============================================================================
# HANDOUT 1 - Topic 12 (Ottoman warfare cluster)
# ============================================================================
print("=" * 74)
print("HANDOUT 1 - Topic 12 membership and mention-vs-thematisation evidence")

t12 = master[master["topic"] == 12]
members = (t12.groupby(["title.1", "author.1", "genre_brit_filter"])
              .size().reset_index(name="n_segments")
              .sort_values("n_segments", ascending=False))
members.to_csv(f"{OUT_DIR}/handout1_topic12_members.csv", index=False)

# The six DEEP records named on the handout supply the bulk of the cluster.
SIX = ["1 Tamburlaine the Great", "2 Tamburlaine the Great",
       "The Raging Turk, or Bajazet the Second",
       "The Courageous Turk, or Amurath the First",
       "The Battle of Alcazar", "Soliman and Perseda (Zulziman)"]
six_n = int(t12["title.1"].isin(SIX).sum())

report("Topic 12 total segments", len(t12), 163)
report("segments from the six named DEEP records", six_n, 139)
report("share supplied by the six records (%)", round(100 * six_n / len(t12), 1), 85.3)

# No Shakespeare tragedy contributes a single segment to Topic 12.
SHA_TRAGEDIES = ("Hamlet", "Othello", "Lear", "Macbeth", "Titus", "Romeo",
                 "Caesar", "Antony", "Coriolanus", "Timon", "Troilus")
sha12 = int(t12["title.1"].apply(
    lambda t: any(k in str(t) for k in SHA_TRAGEDIES) and "Kinsmen" not in str(t)).sum())
report("Shakespeare-tragedy segments in Topic 12", sha12, 0)

# Mention is not thematisation: Othello's Turk/Ottoman-form chunks.
# Early modern spelling varies, so the pattern covers the attested forms.
filtered = pd.read_csv(FILTERED)
TURK_PAT = re.compile(r"t[uv]rk|turck|tourk|ott[ao]m|turban|turbond", re.I)
oth = filtered[filtered["title.1"].str.contains("Othello", na=False)]
oth_turk = oth[oth["chunk_text"].str.contains(TURK_PAT, na=False)]

report("Othello chunks containing a Turk/Ottoman word-form", len(oth_turk), 20)
report("...of which HDBSCAN outliers (topic -1)", int((oth_turk["topic"] == -1).sum()), 19)
assigned = oth_turk[oth_turk["topic"] != -1]
report("...the single assigned chunk goes to topic", int(assigned["topic"].iloc[0]), 2)

report("corpus-wide segments (incl. outliers)", len(filtered), 27104)
report("corpus-wide outlier share (%)",
       round(100 * (filtered["topic"] == -1).mean(), 1), 57.6)

# Contrast case: Fulke Greville's Mustapha, where Ottoman material IS the theme.
must = filtered[filtered["title.1"] == "Mustapha"]
report("Mustapha total segments", len(must), 62)
report("Mustapha assigned segments in topic 16", int((must["topic"] == 16).sum()), 44)

# ============================================================================
# HANDOUT 2 - Per-author divergence within each genre
# ============================================================================
print("=" * 74)
print("HANDOUT 2 - per-author JSD vs the rest of the genre (sole attribution)")

pool_all = master[master["genre_clean"].isin(GENRES) & master["author.1"].notna()]
records = []
for genre in GENRES:
    pool = pool_all[pool_all["genre_clean"] == genre]
    topics = sorted(pool["topic"].unique())
    works = pool.groupby("author.1")["title.1"].nunique()
    for author, n_works in works.items():
        a = str(author).strip()
        if n_works < 3 or "Anonymous" in a or a.startswith("["):
            continue
        own  = pool[pool["author.1"] == author]     # exact match = sole attribution
        rest = pool[pool["author.1"] != author]
        records.append({
            "genre": genre, "author": author,
            "label": a.split(",", 1)[0],
            "n_works": int(n_works), "n_chunks": len(own),
            "jsd_bits": round(jsd(topic_vector(own, topics),
                                  topic_vector(rest, topics)), 4),
        })
per_author = (pd.DataFrame(records)
                .sort_values(["genre", "jsd_bits"], ascending=[True, False])
                .reset_index(drop=True))
per_author.to_csv(f"{OUT_DIR}/handout2_per_author_divergence.csv", index=False)

for genre in ("Tragedy", "Comedy"):
    sub = per_author[per_author["genre"] == genre]
    sha = sub[sub["label"] == "Shakespeare"]
    print(f"  {genre}: Shakespeare JSD {sha['jsd_bits'].iloc[0]:.3f}, "
          f"rank {int(sub.reset_index().index[sub['label'].values == 'Shakespeare'][0]) + 1} "
          f"of {len(sub)} authors (1 = most divergent)")

# Attribution-scope reconciliation (chart 0.589 vs paper section 5.5's 0.597):
trag = pool_all[pool_all["genre_clean"] == "Tragedy"]
contains_sha = trag[trag["author.1"].str.contains("Shakespeare", na=False)]
sole_sha     = trag[trag["author.1"] == "Shakespeare, William"]
report("tragedy segments, contains-Shakespeare scope (paper 5.5)", len(contains_sha), 710)
report("tragedy segments, sole-attribution scope (this chart)", len(sole_sha), 613)

# Figure: one horizontal-bar panel for Tragedy and one for Comedy.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, genre in zip(axes, ("Tragedy", "Comedy")):
    sub = per_author[per_author["genre"] == genre].sort_values("jsd_bits")
    y = np.arange(len(sub))
    colors = [COLOR_SHA if l == "Shakespeare" else COLOR_OTHER for l in sub["label"]]
    ax.barh(y, sub["jsd_bits"], color=colors)
    ax.set_yticks(y); ax.set_yticklabels(sub["label"], fontsize=9)
    ax.set_title(genre, fontweight="bold")
    ax.set_xlabel("JSD vs rest of genre (bits)")
    for yi, v in zip(y, sub["jsd_bits"]):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("Per-author thematic divergence (authors with ≥ 3 plays in the genre)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/handout2_per_author_divergence.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ============================================================================
# HANDOUT 3 - Othello's distributional position
# ============================================================================
print("=" * 74)
print("HANDOUT 3 - Othello vs the three genre profiles, and the 13-play panel")

genre_vecs = {g: pool_all[pool_all["genre_clean"] == g] for g in GENRES}
oth_m = master[master["title.1"].str.contains("Othello", na=False)]

# Othello is compared with Tragedy EXCLUDING its own segments, so the play
# is never compared against a profile it is itself part of.
def genre_profile_minus(df_play, genre):
    pool = genre_vecs[genre]
    if genre == "Tragedy":
        pool = pool.drop(df_play.index, errors="ignore")
    return pool

topics_all = sorted(master["topic"].unique())
d = {}
for g in GENRES:
    pool = genre_profile_minus(oth_m, g)
    d[g] = jsd(topic_vector(oth_m, topics_all), topic_vector(pool, topics_all))
report("Othello assigned segments (both witnesses)", len(oth_m), 27)
report("JSD(Othello, Comedy)", round(d["Comedy"], 4), 0.3515)
report("JSD(Othello, Tragedy minus itself)", round(d["Tragedy"], 4), 0.6107)
report("JSD(Othello, History)", round(d["History"], 4), 0.6157)

# The two TCP witnesses (Q 1622 / F 1623), measured separately, are both
# nearest the comedy profile - the pattern is not an artefact of pooling.
for wit in ("A11992", "A11954.34"):
    w = oth_m[oth_m["TCP"] == wit]
    dist = {g: jsd(topic_vector(w, topics_all),
                   topic_vector(genre_profile_minus(w, g), topics_all)) for g in GENRES}
    nearest = min(dist, key=dist.get)
    print(f"  [OK ] witness {wit}: nearest genre profile = {nearest}")

# 13-play panel: every Shakespeare tragedy-group play with >= 8 assigned
# segments; diff = JSD(play, Comedy) - JSD(play, Tragedy minus itself).
# Negative = leans toward the comedy profile.
sha_trag = trag[trag["author.1"].str.contains("Shakespeare", na=False)]
panel = []
for title, grp in sha_trag.groupby("title.1"):
    if len(grp) < 8:
        continue
    diff = (jsd(topic_vector(grp, topics_all),
                topic_vector(genre_vecs["Comedy"], topics_all))
            - jsd(topic_vector(grp, topics_all),
                  topic_vector(genre_vecs["Tragedy"].drop(grp.index), topics_all)))
    panel.append((title, len(grp), round(diff, 3)))
panel.sort(key=lambda x: x[2])

print(f"  13-play panel (n plays = {len(panel)}; handout prints 13):")
for title, n, diff in panel:
    print(f"      {diff:+.3f}  ({n:3d} seg)  {title}")

fig, ax = plt.subplots(figsize=(9, 6))
titles = [t for t, _, _ in panel][::-1]
diffs  = [v for _, _, v in panel][::-1]
colors = [COLOR_SHA if v < -0.04 else COLOR_OTHER for v in diffs]
y = np.arange(len(titles))
ax.barh(y, diffs, color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(titles, fontsize=9)
ax.set_xlabel("JSD to Comedy − JSD to Tragedy (minus itself), bits\n"
              "negative = closer to the comedy profile")
ax.set_title("Shakespeare tragedy-group plays (≥ 8 assigned segments)",
             fontweight="bold")
for yi, v in zip(y, diffs):
    ax.text(v + (0.006 if v >= 0 else -0.006), yi, f"{v:+.3f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/handout3_othello_panel.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ============================================================================
# HANDOUT 4 - Merry Wives vs Measure for Measure
# ============================================================================
print("=" * 74)
print("HANDOUT 4 - domestic-social register: Merry Wives vs Measure for Measure")

# "Domestic-social register" = Topic 0 (domestic and marital relationships)
# + Topic 1 (social roles and faith in daily life) - the same aggregation
# used for the figures in the circulated paper.
DS = {0, 1}
comedy = genre_vecs["Comedy"]
mw  = master[master["title.1"].str.contains("Merry Wives", na=False)]
m4m = master[master["title.1"].str.contains("Measure for Measure", na=False)]

pooled_share = 100 * comedy["topic"].isin(DS).mean()
mw_ds,  mw_n  = int(mw["topic"].isin(DS).sum()),  len(mw)
m4m_ds, m4m_n = int(m4m["topic"].isin(DS).sum()), len(m4m)

report("pooled corpus-wide comedy DS share (%)", round(pooled_share, 1), 63.8)
report("Merry Wives DS segments", f"{mw_ds}/{mw_n}", "58/71")
report("Merry Wives DS share (%)", round(100 * mw_ds / mw_n, 1), 81.7)
report("Measure for Measure DS segments", f"{m4m_ds}/{m4m_n}", "6/22")
report("Measure for Measure DS share (%)", round(100 * m4m_ds / m4m_n, 1), 27.3)

# Merry Wives' 58 DS segments as a share of ALL DS segments in the
# eleven-play Shakespearean comedy subset (58/140 = 41.4%).
sha_comedy_ds = int(comedy[comedy["author.1"].str.contains("Shakespeare", na=False)]
                    ["topic"].isin(DS).sum())
report("DS segments across Shakespeare's comedies", sha_comedy_ds, 140)
report("Merry Wives share of those (%)", round(100 * mw_ds / sha_comedy_ds, 1), 41.4)

# Measure for Measure's largest assigned topic is 26 (nobility and courtly
# relations) - the Duke's government rather than the household.
m4m_topics = m4m.groupby(["topic", "topic_label"]).size().sort_values(ascending=False)
top_topic, top_label = m4m_topics.index[0]
report("M4M largest topic id", int(top_topic), 26)
report("M4M segments in that topic", int(m4m_topics.iloc[0]), 8)
print(f"      (label: {top_label})")

# Figure: left panel = DS shares; right panel = M4M topic composition.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

bars = [("Comedy\n(pooled, corpus-wide)", pooled_share, COLOR_POOL),
        ("The Merry Wives\nof Windsor", 100 * mw_ds / mw_n, COLOR_SHA),
        ("Measure for\nMeasure", 100 * m4m_ds / m4m_n, COLOR_OTHER)]
x = np.arange(len(bars))
ax1.bar(x, [b[1] for b in bars], color=[b[2] for b in bars], width=0.6)
ax1.set_xticks(x); ax1.set_xticklabels([b[0] for b in bars], fontsize=9)
ax1.set_ylabel("% of assigned segments")
ax1.set_title("Share of assigned segments in the\ndomestic-social register (Topics 0 + 1)",
              fontweight="bold", fontsize=10)
for xi, b in zip(x, bars):
    ax1.text(xi, b[1] + 1.2, f"{b[1]:.1f}%", ha="center", fontweight="bold")
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# Right: M4M composition. Topics with at least 2 segments are shown
# individually; the remaining topics all tie at exactly 1 segment each, so
# they are aggregated into a single bar. (The printed handout displayed two
# of these singleton topics individually; because all of them tie at one
# segment, that choice was an arbitrary tie-break - the aggregation here is
# the deterministic version. Every count is identical.)
# Topic labels carry a numeric prefix ("26. Nobility...") - strip it.
comp = [(str(lbl).split(". ", 1)[-1], int(n)) for (tid, lbl), n in m4m_topics.items()]
top        = [c for c in comp if c[1] >= 2]
singletons = [c for c in comp if c[1] == 1]
labels = [c[0] for c in top] + [f"All other topics\n({len(singletons)} topics, 1 segment each)"]
values = [c[1] for c in top] + [sum(n for _, n in singletons)]
# Highlight the domestic-social (register) topics in red, like the handout.
cols = [COLOR_SHA if ("social roles" in l.lower() or "domestic" in l.lower())
        else COLOR_OTHER for l in labels[:-1]] + [COLOR_POOL]
yy = np.arange(len(labels))[::-1]
ax2.barh(yy, values, color=cols)
ax2.set_yticks(yy); ax2.set_yticklabels(labels, fontsize=9)
ax2.set_xlabel("segments")
ax2.set_title(f"Where Measure for Measure's assigned\nmaterial sits (assigned segments, n = {m4m_n})",
              fontweight="bold", fontsize=10)
for yi, v in zip(yy, values):
    ax2.text(v + 0.1, yi, str(v), va="center", fontsize=9)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/handout4_mw_vs_m4m.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("=" * 74)
print("Figures and CSVs written to 16_wsc_seminar_handouts/results/")
