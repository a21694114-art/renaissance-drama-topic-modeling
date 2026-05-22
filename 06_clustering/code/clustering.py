# ==========================================================
# clustering.py
# Stage: UMAP(5D) + HDBSCAN + UMAP(2D) + interactive plot
# Output folder: /Users/grace/Desktop/new meta/clustering/
# ==========================================================

import os
import numpy as np
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.express as px

# -----------------------------
# 0) Paths (EDIT ONLY THESE if needed)
# -----------------------------
BASE_DIR = "/Users/grace/Desktop/new meta"

EMBED_DIR = os.path.join(BASE_DIR, "embedding")
CHUNK_DIR = os.path.join(BASE_DIR, "chunking")
OUT_DIR = os.path.join(BASE_DIR, "clustering")
os.makedirs(OUT_DIR, exist_ok=True)

# embeddings (from embedding stage)
EMB_NPY = os.path.join(EMBED_DIR, "embeddings_gte_chunk2000.npy")

# chunk table from chunking stage (must align 1-to-1 with embeddings rows)
CHUNKS_CSV = os.path.join(CHUNK_DIR, "chunks_chunk2000.csv")

# outputs
OUT_5D_NPY = os.path.join(OUT_DIR, "reduced_umap_5d.npy")
OUT_2D_NPY = os.path.join(OUT_DIR, "reduced_umap_2d.npy")
OUT_CLUSTER_NPY = os.path.join(OUT_DIR, "clusters_hdbscan.npy")
OUT_TABLE_CSV = os.path.join(OUT_DIR, "cluster_xy_table.csv")
OUT_HTML = os.path.join(OUT_DIR, "interactive_clusters.html")

# -----------------------------
# 1) Params
#    We'll run multiple presets in ONE execution so you can compare:
#      - baseline: your current setting (often high -1)
#      - tuned_less_outliers: usually pulls more points into clusters
#      - tuned_stricter: usually cleaner clusters but more -1
# -----------------------------
RANDOM_STATE = 42

PRESETS = [
    {
        "name": "baseline",
        "umap_5d": {
            "n_components": 5,
            "n_neighbors": 15,
            "min_dist": 0.05,
            "metric": "cosine",
            "random_state": RANDOM_STATE,
        },
        "hdbscan": {
            "min_cluster_size": 30,
            "min_samples": None,           # default ~= min_cluster_size
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        },
        "umap_2d": {
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": RANDOM_STATE,
        },
    },
    {
        # Usually reduces -1 by smoothing neighborhood graph and relaxing density
        "name": "tuned_less_outliers",
        "umap_5d": {
            "n_components": 5,
            "n_neighbors": 35,             # smoother => fewer isolated points
            "min_dist": 0.10,              # slightly less tight packing
            "metric": "cosine",
            "random_state": RANDOM_STATE,
        },
        "hdbscan": {
            "min_cluster_size": 40,        # prefer slightly larger, stabler clusters
            "min_samples": 10,             # key knob: smaller => fewer outliers
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        },
        "umap_2d": {
            "n_components": 2,
            "n_neighbors": 35,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": RANDOM_STATE,
        },
    },
    {
        # Stricter density; often increases -1 but reduces tiny "satellite" clusters
        "name": "tuned_stricter",
        "umap_5d": {
            "n_components": 5,
            "n_neighbors": 20,
            "min_dist": 0.05,
            "metric": "cosine",
            "random_state": RANDOM_STATE,
        },
        "hdbscan": {
            "min_cluster_size": 50,
            "min_samples": None,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        },
        "umap_2d": {
            "n_components": 2,
            "n_neighbors": 20,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": RANDOM_STATE,
        },
    },
]

# -----------------------------
# 2) Load embeddings
# -----------------------------
print("🔎 Loading embeddings:", EMB_NPY)
embeddings = np.load(EMB_NPY)
print("✅ Embeddings shape:", embeddings.shape)

# -----------------------------
# 3) Load chunk table
# -----------------------------
print("🔎 Loading chunk table:", CHUNKS_CSV)
chunks_df = pd.read_csv(CHUNKS_CSV)
print("✅ Chunks table shape:", chunks_df.shape)

if len(chunks_df) != embeddings.shape[0]:
    raise ValueError(
        f"Row mismatch: chunks_df has {len(chunks_df)} rows, "
        f"but embeddings has {embeddings.shape[0]} vectors. "
        f"These must match 1-to-1."
    )

# ensure chunk_id exists
df = chunks_df.copy()
if "chunk_id" not in df.columns:
    if "TCP" in df.columns and "chunk_index" in df.columns:
        df["chunk_id"] = df["TCP"].astype(str) + "_" + df["chunk_index"].astype(str)
    else:
        df["chunk_id"] = np.arange(len(df)).astype(str)

# -----------------------------
# 4) Run presets: UMAP(5D) -> HDBSCAN -> UMAP(2D) -> Save + Plot
# -----------------------------
preferred_hover_cols = [
    "chunk_id",
    "TCP",
    "title.1",
    "author.1",
    "genre_brit_filter",
    "date_first_performance_brit_filter",
    "chunk_index",
    "chunk_len",
]

hover_cols = [c for c in preferred_hover_cols if c in df.columns]

summary_rows = []

for preset in PRESETS:
    name = preset["name"]
    print("\n" + "="*70)
    print(f"🚀 Running preset: {name}")
    print("="*70)

    # outputs per preset
    out_5d_npy = os.path.join(OUT_DIR, f"reduced_umap_5d__{name}.npy")
    out_2d_npy = os.path.join(OUT_DIR, f"reduced_umap_2d__{name}.npy")
    out_cluster_npy = os.path.join(OUT_DIR, f"clusters_hdbscan__{name}.npy")
    out_table_csv = os.path.join(OUT_DIR, f"cluster_xy_table__{name}.csv")
    out_html = os.path.join(OUT_DIR, f"interactive_clusters__{name}.html")

    # 4.1 UMAP -> 5D
    print("🧩 UMAP -> 5D (for clustering) ...")
    umap_model_5d = UMAP(**preset["umap_5d"])
    reduced_5d = umap_model_5d.fit_transform(embeddings)
    np.save(out_5d_npy, reduced_5d)
    print("✅ Saved 5D embeddings:", out_5d_npy)

    # 4.2 HDBSCAN on 5D
    print("🧪 HDBSCAN clustering on 5D ...")
    hdb = HDBSCAN(**preset["hdbscan"])
    hdb.fit(reduced_5d)
    clusters = hdb.labels_.astype(int)
    np.save(out_cluster_npy, clusters)
    print("✅ Saved clusters:", out_cluster_npy)

    n_total = len(clusters)
    n_outliers = int(np.sum(clusters == -1))
    labels = sorted(set(clusters.tolist()))
    n_clusters = len(labels) - (1 if -1 in labels else 0)
    outlier_pct = (n_outliers / n_total) if n_total else 0.0

    vc = pd.Series(clusters).value_counts()
    top10 = vc.head(10)

    print(f"✅ Total points: {n_total}")
    print(f"✅ #Clusters (excluding -1): {n_clusters}")
    print(f"⚠ Outliers (-1): {n_outliers}  ({outlier_pct:.2%})")
    print("📌 Top 10 cluster sizes (label: size):")
    for label, size in top10.items():
        print(f"   {label}: {size}")

    summary_rows.append({
        "preset": name,
        "total_points": n_total,
        "clusters_excl_-1": n_clusters,
        "outliers": n_outliers,
        "outliers_pct": round(outlier_pct * 100, 2),
        "largest_cluster_size": int(vc.iloc[0]) if len(vc) else 0,
    })

    # 4.3 UMAP -> 2D
    print("🗺️ UMAP -> 2D (for visualization) ...")
    umap_model_2d = UMAP(**preset["umap_2d"])
    reduced_2d = umap_model_2d.fit_transform(embeddings)
    np.save(out_2d_npy, reduced_2d)
    print("✅ Saved 2D embeddings:", out_2d_npy)

    # 4.4 Save cluster+xy table
    out_df = df.copy()
    out_df["x"] = reduced_2d[:, 0]
    out_df["y"] = reduced_2d[:, 1]
    out_df["cluster"] = clusters.astype(str)
    out_df.to_csv(out_table_csv, index=False)
    print("✅ Saved cluster+xy table:", out_table_csv)

    # 4.5 Interactive plot
    print("🎨 Building interactive plot ...")
    fig = px.scatter(
        out_df,
        x="x",
        y="y",
        color="cluster",
        hover_data={c: True for c in hover_cols},
        title=f"Interactive HDBSCAN Clusters (UMAP 2D) — {name}",
    )
    fig.update_traces(marker=dict(size=4, opacity=0.6))

    # make -1 light grey
    for trace in fig.data:
        if trace.name == "-1":
            trace.marker.color = "lightgrey"

    fig.update_layout(legend_title_text="Cluster")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.write_html(out_html, include_plotlyjs="cdn")
    print("✅ Saved interactive HTML:", out_html)

# -----------------------------
# 5) Save preset summary
# -----------------------------
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, "preset_summary.csv")
summary_df.to_csv(summary_path, index=False)
print("\n" + "="*70)
print("📄 Preset summary saved:", summary_path)
print(summary_df)
print("🎉 Done.")