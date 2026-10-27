"""
src/tda_mapper.py
-----------------
Topological Data Analysis (TDA) using Kepler Mapper for FinWell.
Implements Chapter 4.5 & Section 5.1.3 of the Black Book:
  • 4.5.1 Application of Kepler Mapper
  • 4.5.2 Dimensionality Reduction with Lenses (PCA Lens)
  • 4.5.3 Simplicial Complexes (Cover + Overlap Clustering)
  • 4.5.4 Visualization and Interpretation (Figure 8: Topological Network)
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import kmapper as km

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
USER_FEATURES_PATH = os.path.join(DATA_DIR, "user_features.csv")
USER_CLUSTERS_PATH = os.path.join(DATA_DIR, "user_clusters.csv")
TOPOLOGICAL_PNG_PATH = os.path.join(OUTPUTS_DIR, "topological_network.png")
TDA_JSON_PATH = os.path.join(OUTPUTS_DIR, "tda_graph.json")

PERSONA_COLORS = {
    0: ("Budget-Conscious",   "#2196F3"),
    1: ("Impulsive Spender",  "#F44336"),
    2: ("Heavy Subscriber",   "#9C27B0"),
    3: ("Luxury Spender",     "#FF9800"),
    4: ("Weekend Shopper",    "#4CAF50"),
}


def build_tda_simplicial_complex(n_cubes: int = 8, perc_overlap: float = 0.35):
    """
    Runs Kepler Mapper algorithm on 500 cohort user features.
    Returns (kmapper_graph, nx_graph, node_metadata, topological_stats).
    """
    if not os.path.exists(USER_FEATURES_PATH) or not os.path.exists(USER_CLUSTERS_PATH):
        raise FileNotFoundError("Missing user_features.csv or user_clusters.csv in data directory.")

    feat_df = pd.read_csv(USER_FEATURES_PATH, index_col="user_id")
    cluster_df = pd.read_csv(USER_CLUSTERS_PATH, index_col="user_id")
    user_ids = feat_df.index.tolist()
    X = feat_df.values

    # 1. Initialize KeplerMapper
    mapper = km.KeplerMapper(verbose=0)

    # 2. Filter / Lens: 2D Principal Component Projection (Section 4.5.2)
    lens = mapper.fit_transform(X, projection=PCA(n_components=2, random_state=42))

    # 3. Cover & Simplicial Clustering (Section 4.5.3)
    cover = km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap)
    clusterer = DBSCAN(eps=0.6, min_samples=3)

    graph = mapper.map(
        lens,
        X,
        cover=cover,
        clusterer=clusterer
    )

    # 4. Convert to NetworkX Graph and construct node metadata
    G = nx.Graph()
    node_meta = {}

    for node_id, member_indices in graph["nodes"].items():
        node_users = [user_ids[i] for i in member_indices]
        node_clusters = [cluster_df.loc[u, "cluster_label"] for u in node_users if u in cluster_df.index]

        if node_clusters:
            # Dominant persona in this simplicial node
            counts = pd.Series(node_clusters).value_counts()
            dom_cluster = int(counts.index[0])
            p_name, p_color = PERSONA_COLORS.get(dom_cluster, ("Mixed", "#64748B"))
        else:
            dom_cluster = 0
            p_name, p_color = ("General", "#64748B")

        avg_spend = float(feat_df.loc[node_users, "avg_monthly_spend"].mean()) if "avg_monthly_spend" in feat_df.columns else 0.5
        avg_savings = float(feat_df.loc[node_users, "savings_ratio"].mean()) if "savings_ratio" in feat_df.columns else 0.5

        node_meta[node_id] = {
            "node_id": node_id,
            "size": len(member_indices),
            "users": node_users,
            "dominant_cluster": dom_cluster,
            "persona_name": p_name,
            "color": p_color,
            "avg_spend": round(avg_spend, 3),
            "avg_savings": round(avg_savings, 3),
        }
        G.add_node(node_id, **node_meta[node_id])

    # Add edges between nodes that share members
    for src, targets in graph["links"].items():
        for tgt in targets:
            if G.has_node(src) and G.has_node(tgt):
                G.add_edge(src, tgt)

    # 5. Compute Topological Invariants (Section 5.1.3)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G) if n_nodes > 0 else 0
    density = nx.density(G) if n_nodes > 1 else 0.0
    avg_clustering = nx.average_clustering(G) if n_nodes > 0 else 0.0

    stats = {
        "n_simplicial_nodes": n_nodes,
        "n_topological_edges": n_edges,
        "connected_components": n_components,
        "graph_density": round(density, 4),
        "avg_clustering_coefficient": round(avg_clustering, 4),
        "lens_method": "Principal Component Analysis (PCA 2D)",
        "clustering_algorithm": "DBSCAN (eps=0.6, min_samples=3)",
        "cover_parameters": {"n_cubes": n_cubes, "perc_overlap": perc_overlap}
    }

    # Save summary json
    with open(TDA_JSON_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    return graph, G, node_meta, stats


def export_static_topological_plot(G, node_meta, save_path=TOPOLOGICAL_PNG_PATH):
    """
    Renders high-resolution Figure 8 (Topological Network) for the academic report.
    """
    if G.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(G, seed=42, k=0.45)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, edge_color="#94A3B8", width=1.5)

    # Draw nodes per persona
    for cid, (p_name, p_color) in PERSONA_COLORS.items():
        c_nodes = [n for n, d in node_meta.items() if d.get("dominant_cluster") == cid]
        if c_nodes:
            sizes = [max(80, node_meta[n]["size"] * 18) for n in c_nodes]
            nx.draw_networkx_nodes(
                G, pos, nodelist=c_nodes, ax=ax,
                node_color=p_color, node_size=sizes,
                alpha=0.85, edgecolors="#FFFFFF", linewidths=1.5,
                label=f"{p_name}"
            )

    ax.set_title(
        "Topological Data Analysis (TDA) — Kepler Mapper Simplicial Complex\n"
        "Figure 8: Behavioral Manifold Across 500 Cohort Users",
        fontsize=13, fontweight="bold", pad=14, color="#0F172A"
    )
    ax.legend(title="Dominant Persona", loc="upper left", frameon=True, facecolor="#FFFFFF")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def get_tda_network_plotly(user_id: str = None) -> go.Figure:
    """
    Builds an interactive Plotly Graph of the Kepler Mapper simplicial network.
    Highlights nodes that contain the active user.
    """
    _, G, node_meta, _ = build_tda_simplicial_complex()

    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="TDA Graph empty or insufficient clusters", showarrow=False)
        return fig

    pos = nx.spring_layout(G, seed=42, k=0.45)

    # 1. Edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color="#CBD5E1"),
        hoverinfo="none",
        mode="lines"
    )

    # 2. Node traces grouped by persona
    fig = go.Figure(data=[edge_trace])

    # Check which nodes contain active user
    active_nodes = []
    if user_id:
        for nid, meta in node_meta.items():
            if user_id in meta["users"]:
                active_nodes.append(nid)

    for cid, (p_name, p_color) in PERSONA_COLORS.items():
        c_nodes = [n for n, d in node_meta.items() if d.get("dominant_cluster") == cid]
        if not c_nodes:
            continue

        xs = [pos[n][0] for n in c_nodes]
        ys = [pos[n][1] for n in c_nodes]
        sizes = [max(12, int(np.sqrt(node_meta[n]["size"]) * 8)) for n in c_nodes]
        hover_texts = [
            f"<b>Node: {n}</b><br>"
            f"Persona: {node_meta[n]['persona_name']}<br>"
            f"Members: {node_meta[n]['size']} users<br>"
            f"Avg Spend: {node_meta[n]['avg_spend']:.2f}<br>"
            f"Avg Savings: {node_meta[n]['avg_savings']:.2f}"
            for n in c_nodes
        ]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            name=p_name,
            text=[f"{node_meta[n]['size']}" for n in c_nodes],
            textposition="middle center",
            textfont=dict(color="#FFFFFF", size=9, family="Inter"),
            hoverinfo="text",
            hovertext=hover_texts,
            marker=dict(
                size=sizes,
                color=p_color,
                line=dict(color="#FFFFFF", width=1.5)
            )
        ))

    # 3. Active User Highlight (Pulsating Golden / Electric Blue Halo)
    if active_nodes:
        u_xs = [pos[n][0] for n in active_nodes]
        u_ys = [pos[n][1] for n in active_nodes]
        fig.add_trace(go.Scatter(
            x=u_xs, y=u_ys,
            mode="markers",
            name="★ Active User Location",
            marker=dict(
                size=[max(28, int(np.sqrt(node_meta[n]["size"]) * 8) + 14) for n in active_nodes],
                color="rgba(20, 110, 245, 0.25)",
                line=dict(color="#146EF5", width=3)
            ),
            hoverinfo="text",
            hovertext=[f"★ {user_id} belongs to this topological cluster ({node_meta[n]['size']} users)" for n in active_nodes]
        ))

    fig.update_layout(
        title=dict(
            text="<b>Topological Data Analysis (Kepler Mapper Simplicial Complex)</b>",
            font=dict(family="Inter", size=15, color="#0F172A")
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Inter", size=10)),
        margin=dict(t=50, b=20, l=20, r=20),
        height=520,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig


if __name__ == "__main__":
    print("Building Kepler Mapper Simplicial Complex...")
    _, G, meta, stats = build_tda_simplicial_complex()
    export_static_topological_plot(G, meta)
    print("TDA Complex Built Successfully!")
    print(f"Nodes: {stats['n_simplicial_nodes']}, Edges: {stats['n_topological_edges']}, Components: {stats['connected_components']}")
    print(f"Figure 8 saved -> {TOPOLOGICAL_PNG_PATH}")
