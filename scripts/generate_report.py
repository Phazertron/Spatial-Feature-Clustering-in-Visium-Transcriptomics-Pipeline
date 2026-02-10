"""
Report generator for spatial clustering pipeline.

Generates comprehensive HTML and PDF reports from session results,
including all plots, metrics, and detailed explanations.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --session run_2026-02-06_14-30-45
    python scripts/generate_report.py --format html
    python scripts/generate_report.py --format pdf
    python scripts/generate_report.py --format both
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import base64

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.session import SessionManager


# ---------------------------------------------------------------------------
# Metric / plot explanation texts
# ---------------------------------------------------------------------------

METRIC_EXPLANATIONS = {
    "silhouette": (
        "The <strong>Silhouette Score</strong> measures how similar each gene is "
        "to its own cluster compared to other clusters. Values range from "
        "<em>&minus;1</em> (wrong cluster) to <em>+1</em> (well-matched). "
        "A score near 0 indicates overlapping clusters."
    ),
    "calinski_harabasz": (
        "The <strong>Calinski-Harabasz Index</strong> (Variance Ratio Criterion) "
        "is the ratio of between-cluster dispersion to within-cluster dispersion. "
        "Higher values indicate denser, better-separated clusters."
    ),
    "davies_bouldin": (
        "The <strong>Davies-Bouldin Index</strong> measures the average similarity "
        "between each cluster and the one most similar to it. "
        "<em>Lower</em> values indicate better separation."
    ),
    "ari": (
        "The <strong>Adjusted Rand Index (ARI)</strong> quantifies the agreement "
        "between two clusterings, adjusted for chance. Values range from "
        "<em>&minus;1</em> (worse than random) through <em>0</em> (random) "
        "to <em>+1</em> (perfect agreement). An ARI &ge; 0.8 is generally "
        "considered strong agreement."
    ),
    "nmi": (
        "The <strong>Normalized Mutual Information (NMI)</strong> measures the "
        "mutual dependence between two clusterings, normalized to [0, 1]. "
        "A value of 1 means the clusterings are identical; 0 means they share "
        "no information."
    ),
    "morans_i": (
        "The <strong>Moran's I</strong> statistic measures spatial autocorrelation "
        "&mdash; the degree to which nearby spots on the Visium slide share "
        "similar gene-expression patterns. Values near <em>+1</em> indicate "
        "strong spatial clustering; near <em>0</em> indicates randomness; near "
        "<em>&minus;1</em> indicates dispersion."
    ),
}

PLOT_EXPLANATIONS = {
    # ---- Notebook 01: Exploration ----
    "nb01_tissue_spots": {
        "title": "Tissue & Spot Layout",
        "caption": (
            "Visium spots overlaid on the H&amp;E tissue image, colored by "
            "total UMI counts per spot. Brighter spots indicate higher "
            "sequencing depth. This overview confirms spot coordinates "
            "and reveals the overall tissue morphology alongside the "
            "expression intensity landscape."
        ),
    },
    "nb01_dist_counts_per_spot": {
        "title": "Distribution of Total Counts per Spot",
        "caption": (
            "Histogram of the total UMI (unique molecular identifier) counts "
            "per spot. Spots with very low counts may indicate empty or "
            "low-quality capture areas. A unimodal distribution with a long "
            "right tail is typical for Visium data."
        ),
    },
    "nb01_dist_counts_per_gene": {
        "title": "Distribution of Total Counts per Gene",
        "caption": (
            "Histogram of the total counts aggregated per gene across all "
            "spots. Most genes have very low total counts; a small number of "
            "highly expressed genes dominate. This motivates filtering to the "
            "top spatially variable genes."
        ),
    },
    "nb01_dist_genes_per_spot": {
        "title": "Distribution of Detected Genes per Spot",
        "caption": (
            "Histogram of the number of distinct genes detected (count &gt; 0) "
            "per spot. This is a key QC metric &mdash; spots with unusually "
            "few detected genes may lie outside the tissue or suffer from "
            "capture failure."
        ),
    },
    # ---- Notebook 02: Baseline ----
    "nb02_baseline_similarity_matrices": {
        "title": "Baseline Similarity Matrices (Pearson / Spearman / Cosine)",
        "caption": (
            "Heatmaps of the three expression-only gene-gene similarity "
            "matrices. <strong>Pearson</strong> captures linear correlation, "
            "<strong>Spearman</strong> captures rank-order correlation, and "
            "<strong>Cosine</strong> measures directional similarity. Block "
            "structure indicates gene modules detectable from expression alone."
        ),
    },
    # ---- Notebook 03: Weighted similarity ----
    "nb03_weighted_similarity_heatmap": {
        "title": "Spatial Weighted Similarity Matrix (NB03)",
        "caption": (
            "Heatmap of the combined similarity matrix computed with weights "
            "&alpha;&middot;Expr + &beta;&middot;Spatial + &gamma;&middot;MoG "
            "in notebook 03. This is the primary similarity used for the "
            "initial weighted clustering."
        ),
    },
    # ---- Notebook 04: Multi-view ----
    "nb04_ari_nmi_heatmaps": {
        "title": "ARI / NMI Inter-View Comparison (NB04)",
        "caption": (
            "Pairwise comparison of the four clustering views (expression, "
            "spatial, MoG, weighted) using ARI (left) and NMI (right). "
            "High off-diagonal values mean the two views largely agree on "
            "gene groupings; lower values reveal genes reclassified when "
            "spatial or MoG information is introduced."
        ),
    },
    # ---- Notebook 05: Final plots (saved by name) ----
    "all_similarity_matrices": {
        "title": "Similarity Matrices Overview (Publication)",
        "caption": (
            "Side-by-side heatmaps of the four gene-gene similarity matrices "
            "used in multi-view clustering: <strong>Expression</strong> "
            "(Pearson on raw counts), <strong>Spatial</strong> (after mean "
            "filter), <strong>MoG</strong> (binarized), and "
            "<strong>Weighted</strong> (&alpha;&middot;Expr + &beta;&middot;Spatial "
            "+ &gamma;&middot;MoG). Block-diagonal structure indicates clear "
            "gene modules."
        ),
    },
    "ari_nmi_matrices": {
        "title": "ARI / NMI Matrices (Publication)",
        "caption": (
            "Publication-quality pairwise comparison matrices. ARI (left) "
            "and NMI (right) between all four clustering views. Values close "
            "to 1.0 confirm that the gene modules are robust across "
            "representations."
        ),
    },
    "spatial_coherence_bar": {
        "title": "Spatial Coherence per Cluster (Moran's I)",
        "caption": (
            "Bar chart of the average <strong>Moran's I</strong> for each gene "
            "cluster. A tall bar means the genes in that cluster have "
            "spatially coherent expression profiles across the Visium tissue, "
            "validating that the cluster captures a biologically meaningful "
            "spatial pattern."
        ),
    },
    "nb05_cluster_spatial_profiles": {
        "title": "Cluster Spatial Profiles",
        "caption": (
            "Spatial distribution of gene clusters on the tissue. Each panel "
            "shows one cluster's average expression across all Visium spots, "
            "overlaid on the H&amp;E tissue image. Moran's I quantifies "
            "spatial coherence &mdash; higher values indicate the gene module "
            "is spatially structured rather than randomly distributed."
        ),
    },
    "weighted_similarity_matrix": {
        "title": "Weighted Similarity Matrix (Publication)",
        "caption": (
            "Detailed view of the final weighted similarity matrix that "
            "combines expression, spatial, and MoG components with weights "
            "&alpha;, &beta;, &gamma;. Genes are ordered by cluster assignment; "
            "the visible block structure confirms well-defined gene modules."
        ),
    },
    # ---- Notebook 06: Sensitivity analysis ----
    "resolution_optimization": {
        "title": "Resolution Optimization",
        "caption": (
            "Grid search over the Louvain/Leiden resolution parameter. "
            "The left panel shows <strong>Silhouette score vs. resolution</strong> "
            "&mdash; the peak identifies the resolution that produces the most "
            "internally cohesive and well-separated clusters. The right panel "
            "shows how the <strong>number of clusters</strong> grows with "
            "resolution."
        ),
    },
    "resolution_comparison": {
        "title": "Optimized vs. Fixed Resolution",
        "caption": (
            "Comparison of clustering outcomes when using the automatically "
            "optimized resolution versus the default fixed value (1.0). "
            "If both produce similar ARI / cluster counts, the default "
            "resolution is already near-optimal for this dataset."
        ),
    },
    "weight_sensitivity_heatmaps": {
        "title": "Weight Sensitivity Analysis",
        "caption": (
            "Heatmaps showing how cluster quality varies across different "
            "combinations of the three similarity weights: "
            "<strong>&alpha;</strong> (expression), <strong>&beta;</strong> "
            "(spatial), and <strong>&gamma;</strong> (MoG). The left heatmap "
            "displays ARI relative to the baseline, while the right shows "
            "Silhouette score. Regions of uniformly high ARI indicate "
            "<em>robust</em> weight ranges where the clustering is stable."
        ),
    },
    "pca_variance_explained": {
        "title": "PCA Variance Explained",
        "caption": (
            "Left: <strong>scree plot</strong> showing individual variance "
            "explained by each principal component &mdash; the rapid drop-off "
            "identifies the intrinsic dimensionality of the data. "
            "Right: <strong>cumulative variance</strong> with the 95%% "
            "threshold (red) and the elbow point (orange). Retaining &ge; 95%% "
            "of variance ensures downstream neighbor computation operates on "
            "a faithful low-dimensional representation."
        ),
    },
}

# Pattern-based captions for dynamically named plots (prefix -> description)
PLOT_PATTERN_CAPTIONS = [
    ("nb01_gene_diag_", {
        "title_prefix": "Gene Diagnostic (Exploration)",
        "caption": (
            "Full diagnostic plot for a randomly sampled gene from notebook 01. "
            "Shows: raw spatial expression, mean-filtered spatial map, "
            "non-zero value histogram, ordered CDF, GMM fit on filtered "
            "data, and MoG-binarized spatial map. This panel reveals "
            "whether the gene has a clear spatial domain (visible in the "
            "MoG panel) or is diffusely expressed."
        ),
    }),
    ("nb01_marker_", {
        "title_prefix": "Known Marker Gene",
        "caption": (
            "Diagnostic plot for a known marker gene (e.g., MOBP for "
            "oligodendrocytes in DLPFC). The spatial pattern should match "
            "known tissue layer anatomy, serving as a sanity check for "
            "data quality and spatial alignment."
        ),
    }),
    ("nb02_baseline_cluster_", {
        "title_prefix": "Baseline Cluster Representative",
        "caption": (
            "Diagnostic plot for a representative gene from a baseline "
            "(expression-only Pearson) cluster. The spatial pattern "
            "illustrates the type of expression structure captured "
            "without any spatial filtering."
        ),
    }),
    ("nb03_weighted_cluster_", {
        "title_prefix": "Weighted Cluster Representative",
        "caption": (
            "Diagnostic plot for a representative gene from the spatially "
            "weighted clustering (NB03). Compared to baseline, the MoG "
            "panel should show cleaner spatial domains thanks to the "
            "inclusion of spatial and MoG similarity components."
        ),
    }),
    ("nb04_view_", {
        "title_prefix": "Multi-View Representative",
        "caption": (
            "Diagnostic plot for a representative gene from one of the "
            "four multi-view clusterings (expression, spatial, MoG, or "
            "weighted). Comparing across views reveals how different "
            "similarity representations emphasize different spatial patterns."
        ),
    }),
    ("nb04_changed_gene_", {
        "title_prefix": "View-Switching Gene (NB04)",
        "caption": (
            "Diagnostic plot for a gene that <em>changed cluster</em> "
            "between the two most different views. These genes sit at "
            "the boundary between modules and are biologically interesting "
            "&mdash; their grouping depends on whether spatial context "
            "is considered."
        ),
    }),
    ("nb05_final_cluster_", {
        "title_prefix": "Final Cluster Representative",
        "caption": (
            "Publication-quality diagnostic plot for a representative "
            "gene from the final weighted clustering. The six panels "
            "provide a complete picture of the gene's expression "
            "distribution and spatial structure."
        ),
    }),
    ("nb05_changed_gene_", {
        "title_prefix": "View-Switching Gene (Final)",
        "caption": (
            "Diagnostic plot for a gene that changed cluster between the "
            "two most different views, generated during the final "
            "analysis. These genes are candidates for further biological "
            "investigation."
        ),
    }),
]

SECTION_EXPLANATIONS = {
    "executive_summary": (
        "This report summarizes the results of a full spatial feature "
        "clustering pipeline run on 10x Genomics Visium transcriptomics data. "
        "The pipeline clusters <strong>genes</strong> (not spots/cells) into "
        "co-expression modules by selecting spatially variable genes, computing "
        "multiple similarity representations (expression, spatial, MoG, weighted), "
        "clustering genes under each representation, and evaluating cluster "
        "quality using both internal metrics and spatial coherence."
    ),
    "optimal_parameters": (
        "The parameters below were identified through systematic grid search "
        "over the weight space (&alpha; + &beta; + &gamma; = 1) and resolution "
        "range. The <em>optimal</em> combination maximizes the Silhouette "
        "score of the weighted clustering."
    ),
    "baseline_metrics": (
        "Baseline clustering uses <strong>expression-only</strong> similarity "
        "matrices (Pearson, Spearman, Cosine) without any spatial filtering. "
        "These metrics serve as the reference point against which spatially "
        "informed clusterings are compared."
    ),
    "multiview_comparison": (
        "The multi-view analysis clusters genes under four different "
        "similarity representations and then measures pairwise agreement. "
        "High ARI/NMI between views confirms that the core gene modules "
        "are robust; divergences highlight genes whose grouping is "
        "sensitive to spatial context."
    ),
    "cluster_summary": (
        "Summary of the final weighted clustering. <strong>Size</strong> is "
        "the number of genes assigned to each cluster. "
        "<strong>Spatial Coherence (Moran's I)</strong> quantifies how "
        "spatially structured the average expression profile of each cluster "
        "is on the Visium tissue."
    ),
    "sensitivity": (
        "Sensitivity analysis verifies that the chosen parameters are robust. "
        "If small changes to weights or resolution cause large shifts in "
        "cluster assignments (low ARI), the result is fragile. Stable, high "
        "ARI across a wide parameter range indicates a trustworthy clustering."
    ),
    "visualizations": (
        "All visualizations generated during the pipeline run are collected "
        "below. Each figure is accompanied by an explanation of what it "
        "shows and how to interpret it."
    ),
}

# ---------------------------------------------------------------------------
# HTML template (dark theme)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spatial Feature Clustering Report &mdash; {sample_id} &mdash; {session_id}</title>
    <style>
        /* ===== Reset & Base ===== */
        *, *::before, *::after {{
            margin: 0; padding: 0; box-sizing: border-box;
        }}
        html {{ scroll-behavior: smooth; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.65;
            color: #cfd8dc;
            background: #0d1117;
            padding: 24px;
        }}

        /* ===== Layout ===== */
        .container {{
            max-width: 1160px;
            margin: 0 auto;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 48px 44px;
            box-shadow: 0 8px 32px rgba(0,0,0,.45);
        }}

        /* ===== Typography ===== */
        h1 {{
            color: #e6edf3;
            font-size: 1.85rem;
            border-bottom: 2px solid #58a6ff;
            padding-bottom: 12px;
            margin-bottom: 32px;
        }}
        h2 {{
            color: #e6edf3;
            font-size: 1.35rem;
            margin-top: 44px;
            margin-bottom: 14px;
            border-left: 4px solid #58a6ff;
            padding-left: 14px;
        }}
        h3 {{
            color: #8b949e;
            font-size: 1.08rem;
            margin-top: 24px;
            margin-bottom: 10px;
        }}
        p, li {{ color: #b1bac4; }}
        a {{ color: #58a6ff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}

        /* ===== Section explanation blurbs ===== */
        .section-info {{
            background: #1c2128;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 22px;
            font-size: 0.92rem;
            line-height: 1.55;
            color: #8b949e;
        }}

        /* ===== Metadata bar ===== */
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 10px;
            background: #1c2128;
            padding: 18px 22px;
            border-radius: 8px;
            border: 1px solid #30363d;
            margin-bottom: 30px;
        }}
        .metadata-item {{
            font-size: 0.88rem;
        }}
        .metadata-label {{
            font-weight: 600;
            color: #8b949e;
            margin-right: 6px;
        }}
        .metadata-value {{
            color: #cfd8dc;
        }}

        /* ===== Tables ===== */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 0.9rem;
        }}
        thead th {{
            background: #21262d;
            color: #e6edf3;
            padding: 10px 14px;
            text-align: left;
            border-bottom: 2px solid #30363d;
            font-weight: 600;
        }}
        tbody td {{
            padding: 8px 14px;
            border-bottom: 1px solid #21262d;
            color: #b1bac4;
        }}
        tbody tr:hover {{
            background: #1c2128;
        }}

        /* ===== Metric cards (compact grid) ===== */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
            gap: 12px;
            margin: 14px 0;
        }}
        .metric-card {{
            background: #1c2128;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 12px 14px;
            text-align: center;
        }}
        .metric-card .label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #8b949e;
            margin-bottom: 4px;
        }}
        .metric-card .value {{
            font-size: 1.2rem;
            font-weight: 700;
            color: #58a6ff;
        }}
        .metric-card .value.good {{ color: #3fb950; }}
        .metric-card .value.warn {{ color: #d29922; }}
        .metric-card .value.bad  {{ color: #f85149; }}

        /* ===== Optimal parameters banner ===== */
        .optimal-banner {{
            background: linear-gradient(135deg, #0d2818 0%, #1c2128 100%);
            border: 1px solid #238636;
            border-radius: 8px;
            padding: 22px 26px;
            margin: 22px 0;
        }}
        .optimal-banner h3 {{
            color: #3fb950;
            margin: 0 0 6px 0;
            font-size: 1.05rem;
        }}
        .optimal-banner .params-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 14px;
        }}
        .optimal-banner .param-item {{
            background: #0d1117;
            border: 1px solid #238636;
            border-radius: 6px;
            padding: 10px 12px;
            text-align: center;
        }}
        .optimal-banner .param-item .label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            color: #8b949e;
            margin-bottom: 2px;
        }}
        .optimal-banner .param-item .value {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #3fb950;
        }}

        /* ===== Note / warning box ===== */
        .note-box {{
            background: #1c1d00;
            border: 1px solid #9e6a03;
            border-radius: 6px;
            padding: 14px 18px;
            margin: 18px 0;
            font-size: 0.9rem;
            color: #d29922;
        }}

        /* ===== Config block ===== */
        .config-block {{
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px 20px;
            margin: 14px 0;
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
            font-size: 0.82rem;
            line-height: 1.7;
            color: #8b949e;
            white-space: pre-wrap;
            overflow-x: auto;
        }}

        /* ===== Status badges ===== */
        .badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .badge-success {{ background: #238636; color: #fff; }}
        .badge-fail    {{ background: #da3633; color: #fff; }}

        /* ===== Plot containers ===== */
        .plot-section {{
            background: #1c2128;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 22px;
            margin: 24px 0;
        }}
        .plot-section h3 {{
            color: #e6edf3;
            margin: 0 0 8px 0;
        }}
        .plot-section .plot-explanation {{
            font-size: 0.88rem;
            color: #8b949e;
            margin-bottom: 16px;
            line-height: 1.55;
        }}
        .plot-section img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid #30363d;
            display: block;
            margin: 0 auto;
        }}

        /* ===== CSV data tables (ARI/NMI matrices) ===== */
        .data-table-wrap {{
            overflow-x: auto;
            margin: 14px 0;
        }}
        .data-table-wrap table {{
            min-width: 400px;
        }}

        /* ===== Inline code ===== */
        code {{
            background: #0d1117;
            border: 1px solid #30363d;
            padding: 1px 6px;
            border-radius: 4px;
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
            font-size: 0.88em;
            color: #79c0ff;
        }}

        /* ===== Footer ===== */
        .footer {{
            margin-top: 48px;
            padding-top: 18px;
            border-top: 1px solid #21262d;
            text-align: center;
            color: #484f58;
            font-size: 0.82rem;
        }}

        /* ===== Print / PDF overrides ===== */
        @media print {{
            body {{ background: #0d1117; -webkit-print-color-adjust: exact; }}
            .container {{ box-shadow: none; border: none; }}
            .plot-section {{ break-inside: avoid; }}
        }}
    </style>
</head>
<body>
<div class="container">

    <h1>Spatial Feature Clustering &mdash; {sample_id}</h1>

    <div class="metadata">
        <div class="metadata-item">
            <span class="metadata-label">Session:</span>
            <span class="metadata-value"><code>{session_id}</code></span>
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Profile:</span>
            <span class="metadata-value"><code>{profile}</code></span>
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Started:</span>
            <span class="metadata-value">{start_time}</span>
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Finished:</span>
            <span class="metadata-value">{end_time}</span>
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Description:</span>
            <span class="metadata-value">{description}</span>
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Sample:</span>
            <span class="metadata-value"><code>{sample_id}</code></span>
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Dataset:</span>
            <span class="metadata-value"><code>{dataset_path}</code></span>
        </div>
    </div>

    <!-- Sample Overview & Analysis Note -->
    {sample_overview_section}

    <!-- Executive Summary -->
    <h2>Executive Summary</h2>
    <div class="section-info">{executive_summary_explanation}</div>
    {executive_summary}

    <!-- Optimal Parameters -->
    {optimal_params_section}

    <!-- Configuration -->
    <h2>Configuration</h2>
    <div class="config-block">{config_display}</div>

    <!-- Execution Status -->
    <h2>Execution Status</h2>
    {execution_status}

    <!-- Baseline Metrics -->
    <h2>Baseline Metrics</h2>
    <div class="section-info">{baseline_explanation}</div>
    {baseline_section}

    <!-- Multi-View Comparison -->
    <h2>Multi-View Comparison</h2>
    <div class="section-info">{multiview_explanation}</div>
    {multiview_section}

    <!-- Cluster Summary -->
    <h2>Cluster Summary</h2>
    <div class="section-info">{cluster_summary_explanation}</div>
    {cluster_summary_section}

    <!-- Sensitivity Analysis -->
    {sensitivity_section}

    <!-- Numpy Array Summaries -->
    {array_summaries_section}

    <!-- Visualizations -->
    <h2>Visualizations</h2>
    <div class="section-info">{visualizations_explanation}</div>
    {plots_section}

    <!-- Detailed Results -->
    <h2>Session Artifacts</h2>
    {detailed_results}

    <div class="footer">
        Generated on {generation_time}<br>
        Spatial Feature Clustering in Visium Transcriptomics</div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    session_id: Optional[str] = None,
    output_format: str = "both",
    output_path: Optional[Path] = None
):
    """
    Generate analysis report from session results.

    Parameters
    ----------
    session_id : str, optional
        Session ID to generate report for. If None, uses current session.
    output_format : str
        Output format: "html", "pdf", or "both".
    output_path : Path, optional
        Custom output path. If None, saves to session directory.

    Returns
    -------
    Path or tuple of Path
        Path(s) to generated report(s).
    """
    # Get session
    if session_id:
        run_dir = Path(SessionManager.RUNS_DIR) / session_id
        if not run_dir.exists():
            raise ValueError(f"Session {session_id} not found")

        with open(run_dir / "session.json", "r") as f:
            metadata = json.load(f)

        session = SessionManager(session_id, run_dir, metadata["config"])
    else:
        session = SessionManager.get_current_session()
        if session is None:
            raise RuntimeError(
                "No active session found. Specify session_id or run a pipeline first."
            )

    print(f"Generating report for session: {session.session_id}")

    # Load session metadata
    with open(session.run_dir / "session.json", "r") as f:
        metadata = json.load(f)

    # Collect data for report
    report_data = _collect_report_data(session, metadata)

    # Generate HTML report
    html_content = None
    html_path = None
    if output_format in ["html", "both"]:
        html_content = _generate_html_report(report_data)

        if output_path:
            html_path = output_path
        else:
            html_path = session.run_dir / "report.html"

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"HTML report saved to: {html_path}")

    # Generate PDF report
    pdf_path = None
    if output_format in ["pdf", "both"]:
        try:
            pdf_path = _generate_pdf_report(
                session, html_content if html_content else None
            )
            print(f"PDF report saved to: {pdf_path}")
        except ImportError:
            print("PDF generation requires 'weasyprint'. Install with:")
            print("   pip install weasyprint")
            print("   Skipping PDF generation.")
        except OSError as e:
            # WeasyPrint on Windows requires GTK3/Pango native libraries.
            # This is a known limitation on Windows without MSYS2/GTK3.
            print(f"PDF generation skipped (missing native library): {e}")
            print("   On Windows, WeasyPrint needs GTK3 runtime libraries.")
            print("   Install via MSYS2: pacman -S mingw-w64-x86_64-pango")
            print("   Or use --format html to generate HTML reports only.")

    if output_format == "both":
        return html_path, pdf_path
    elif output_format == "html":
        return html_path
    else:
        return pdf_path


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect_report_data(
    session: SessionManager, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Collect all data needed for report generation."""
    import numpy as np
    import pandas as pd

    import re

    # Extract dataset path and sample ID
    dataset_path = metadata.get(
        "dataset_path",
        metadata.get("config", {}).get("dataset_path", "data/DLPFC-151673")
    )
    sample_match = re.search(r'DLPFC-(\d+)', str(dataset_path))
    sample_id = sample_match.group(0) if sample_match else Path(dataset_path).name

    # Embed raw H&E tissue image for the report header
    tissue_image_b64 = None
    tissue_image_path = Path(dataset_path) / "spatial" / "tissue_hires_image.png"
    if tissue_image_path.exists():
        with open(tissue_image_path, "rb") as img_f:
            tissue_image_b64 = base64.b64encode(img_f.read()).decode()

    data: Dict[str, Any] = {
        "session_id": session.session_id,
        "sample_id": sample_id,
        "dataset_path": str(dataset_path),
        "tissue_image_b64": tissue_image_b64,
        "profile": metadata.get("profile", "unknown"),
        "start_time": metadata.get("start_time", "N/A"),
        "end_time": metadata.get("end_time", "N/A"),
        "description": metadata.get("description", ""),
        "config": metadata.get("config", {}),
        "execution_summary": None,
        # Structured metric buckets
        "baseline_metrics": None,      # DataFrame from baseline_metrics.csv
        "ari_matrix": None,            # DataFrame from ari_matrix.csv
        "nmi_matrix": None,            # DataFrame from nmi_matrix.csv
        "cluster_summary": None,       # DataFrame from cluster_summary.csv
        "array_summaries": {},         # dict[name -> summary dict]
        "sensitivity": None,           # dict from sensitivity_results.npy
        "optimal_parameters": None,
        "plots": [],
    }

    # -- Execution summary --
    summary_file = session.run_dir / "execution_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            data["execution_summary"] = json.load(f)

    # -- CSV tables --
    csv_map = {
        "baseline_metrics": "baseline_metrics.csv",
        "ari_matrix": "ari_matrix.csv",
        "nmi_matrix": "nmi_matrix.csv",
        "cluster_summary": "cluster_summary.csv",
    }
    for key, filename in csv_map.items():
        csv_path = session.metrics_dir / filename
        if csv_path.exists():
            try:
                data[key] = pd.read_csv(csv_path, index_col=0)
            except Exception:
                pass

    # -- Numpy arrays (summary statistics only) --
    for npy_file in sorted(session.metrics_dir.glob("*.npy")):
        name = npy_file.stem
        # Skip sensitivity — handled separately
        if name in ("sensitivity_results", "optimized_resolutions"):
            continue
        try:
            arr = np.load(npy_file, allow_pickle=True)
            if arr.ndim == 0:
                # Scalar or dict stored via allow_pickle
                data["array_summaries"][name] = {
                    "shape": "(scalar)", "dtype": str(arr.dtype)
                }
            elif arr.ndim == 1:
                data["array_summaries"][name] = {
                    "shape": str(arr.shape),
                    "dtype": str(arr.dtype),
                    "min": f"{float(np.nanmin(arr)):.4f}" if arr.size else "N/A",
                    "max": f"{float(np.nanmax(arr)):.4f}" if arr.size else "N/A",
                    "mean": f"{float(np.nanmean(arr)):.4f}" if arr.size else "N/A",
                    "unique": str(len(np.unique(arr))) if arr.size else "N/A",
                }
            elif arr.ndim == 2:
                data["array_summaries"][name] = {
                    "shape": f"{arr.shape[0]} x {arr.shape[1]}",
                    "dtype": str(arr.dtype),
                    "min": f"{float(np.nanmin(arr)):.4f}",
                    "max": f"{float(np.nanmax(arr)):.4f}",
                    "mean": f"{float(np.nanmean(arr)):.4f}",
                }
            else:
                data["array_summaries"][name] = {
                    "shape": str(arr.shape), "dtype": str(arr.dtype)
                }
        except Exception:
            pass

    # -- Sensitivity results --
    sensitivity_file = session.metrics_dir / "sensitivity_results.npy"
    opt_res_file = session.metrics_dir / "optimized_resolutions.npy"

    if sensitivity_file.exists():
        try:
            sens = np.load(sensitivity_file, allow_pickle=True).item()
            data["sensitivity"] = sens

            combos = sens["combinations"]
            sil = sens["silhouette_scores"]
            best_idx = int(np.argmax(sil))
            best_w = combos[best_idx]

            optimal = {
                "alpha": float(best_w[0]),
                "beta": float(best_w[1]),
                "gamma": float(best_w[2]),
                "silhouette": float(sil[best_idx]),
            }

            if opt_res_file.exists():
                opt_res = np.load(opt_res_file, allow_pickle=True).item()
                if "weighted" in opt_res:
                    optimal["resolution"] = float(opt_res["weighted"])

            data["optimal_parameters"] = optimal
        except Exception:
            pass

    # -- Plots --
    # Group plots by notebook/section so the report reads logically.
    # Prefix ordering defines the section sequence.
    prefix_order = [
        "nb01_",            # Exploration
        "nb02_",            # Baseline
        "nb03_",            # Weighted similarity
        "nb04_",            # Multi-view
        "all_similarity",   # NB05 publication
        "ari_nmi_",         # NB05 publication
        "spatial_coherence", # NB05 publication
        "weighted_similarity_matrix", # NB05 publication
        "nb05_",            # NB05 diagnostics
        "resolution_",      # NB06 sensitivity
        "weight_sensitivity", # NB06 sensitivity
        "pca_",             # NB06 PCA
    ]

    all_pngs = sorted(session.plots_dir.glob("*.png"))
    added: set = set()

    for prefix in prefix_order:
        for p in all_pngs:
            if p.stem.startswith(prefix) and p.stem not in added:
                data["plots"].append({"name": p.stem, "path": p})
                added.add(p.stem)
    # Append any remaining plots not matched by prefix
    for p in all_pngs:
        if p.stem not in added:
            data["plots"].append({"name": p.stem, "path": p})

    return data


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _generate_html_report(data: Dict[str, Any]) -> str:
    """Generate the full HTML report string."""
    return HTML_TEMPLATE.format(
        session_id=data["session_id"],
        sample_id=data.get("sample_id", "Unknown"),
        dataset_path=data.get("dataset_path", "N/A"),
        profile=data["profile"],
        start_time=data["start_time"],
        end_time=data["end_time"],
        description=data["description"] or "N/A",
        # Sample overview (tissue image + gene-clustering note)
        sample_overview_section=_build_sample_overview_section(data),
        # Explanation blurbs
        executive_summary_explanation=SECTION_EXPLANATIONS["executive_summary"],
        baseline_explanation=SECTION_EXPLANATIONS["baseline_metrics"],
        multiview_explanation=SECTION_EXPLANATIONS["multiview_comparison"],
        cluster_summary_explanation=SECTION_EXPLANATIONS["cluster_summary"],
        visualizations_explanation=SECTION_EXPLANATIONS["visualizations"],
        # Dynamic sections
        executive_summary=_build_executive_summary(data),
        optimal_params_section=_build_optimal_parameters_section(data),
        config_display=_format_config(data["config"]),
        execution_status=_build_execution_status(data),
        baseline_section=_build_baseline_section(data),
        multiview_section=_build_multiview_section(data),
        cluster_summary_section=_build_cluster_summary_section(data),
        sensitivity_section=_build_sensitivity_section(data),
        array_summaries_section=_build_array_summaries_section(data),
        plots_section=_build_plots_section(data),
        detailed_results=_build_detailed_results(data),
        generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_executive_summary(data: Dict[str, Any]) -> str:
    summary = data.get("execution_summary")
    if not summary:
        return "<p>Execution summary not available.</p>"

    total = summary.get("total", 0)
    succeeded = summary.get("succeeded", 0)
    failed = summary.get("failed", 0)

    badge = "badge-success" if failed == 0 else "badge-fail"
    status_text = "ALL PASSED" if failed == 0 else f"{failed} FAILED"

    return (
        f'<p>Pipeline execution completed: '
        f'<strong>{succeeded}/{total}</strong> notebooks succeeded. '
        f'<span class="badge {badge}">{status_text}</span></p>'
    )


def _build_optimal_parameters_section(data: Dict[str, Any]) -> str:
    params = data.get("optimal_parameters")
    if not params:
        return (
            '<div class="note-box">'
            "Optimal parameters not available. Run sensitivity analysis "
            "(notebook 06) to determine data-driven parameters."
            "</div>"
        )

    html = (
        '<div class="optimal-banner">'
        '<h3>Optimal Parameters (Sensitivity Analysis)</h3>'
        f'<div class="section-info" style="border:none;background:transparent;'
        f'padding:4px 0;">{SECTION_EXPLANATIONS["optimal_parameters"]}</div>'
        '<div class="params-grid">'
    )

    items = [
        ("&alpha; (Expression)", f"{params['alpha']:.2f}"),
        ("&beta; (Spatial)", f"{params['beta']:.2f}"),
        ("&gamma; (MoG)", f"{params['gamma']:.2f}"),
        ("Silhouette", f"{params['silhouette']:.3f}"),
    ]
    if "resolution" in params:
        items.append(("Resolution", f"{params['resolution']:.2f}"))

    for label, value in items:
        html += (
            '<div class="param-item">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            '</div>'
        )

    html += '</div></div>'
    return html


def _format_config(config: Dict[str, Any], indent: int = 0) -> str:
    lines: List[str] = []
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{'  ' * indent}{key}:")
            lines.append(_format_config(value, indent + 1))
        else:
            lines.append(f"{'  ' * indent}{key}: {value}")
    return "\n".join(lines)


def _build_execution_status(data: Dict[str, Any]) -> str:
    summary = data.get("execution_summary")
    if not summary:
        return "<p>No execution data available.</p>"

    rows = ""
    for nb in summary.get("notebooks", []):
        name = nb.get("notebook", "Unknown")
        status = nb.get("status", "unknown")
        badge = "badge-success" if status == "success" else "badge-fail"
        rows += (
            f'<tr><td><code>{name}</code></td>'
            f'<td><span class="badge {badge}">{status.upper()}</span></td></tr>'
        )

    return (
        "<table><thead><tr><th>Notebook</th><th>Status</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _build_sample_overview_section(data: Dict[str, Any]) -> str:
    """Build the sample overview with raw H&E image and gene-clustering note."""
    parts = []

    tissue_b64 = data.get("tissue_image_b64")
    sample_id = data.get("sample_id", "Unknown")

    if tissue_b64:
        parts.append(
            '<div class="plot-section">'
            f'<h3>Sample Tissue &mdash; {sample_id}</h3>'
            '<div class="plot-explanation">'
            'H&amp;E stained tissue section from the Visium slide. This image '
            'identifies the analyzed sample and provides anatomical context '
            'for interpreting the spatial expression patterns discovered by '
            'the clustering pipeline.'
            '</div>'
            f'<img src="data:image/png;base64,{tissue_b64}" '
            'alt="H&amp;E tissue image" style="max-width:600px;">'
            '</div>'
        )

    parts.append(
        '<div class="note-box" style="background:#1c2128;border:1px solid #58a6ff;'
        'color:#8b949e;">'
        '<strong style="color:#58a6ff;">Note &mdash; Gene-Level Clustering</strong>'
        '<br>'
        'This pipeline clusters <strong style="color:#cfd8dc;">genes</strong> into '
        'co-expression modules based on their spatial expression patterns across all '
        'Visium spots. This is fundamentally different from 10x Genomics Space Ranger, '
        'which clusters <strong style="color:#cfd8dc;">spots</strong> (tissue regions / '
        'cell types). The gene modules identified here represent groups of genes with '
        'similar spatial expression structure, not tissue domains.'
        '</div>'
    )

    return "\n".join(parts)


def _build_baseline_section(data: Dict[str, Any]) -> str:
    """Render baseline metrics table with per-metric explanations."""
    df = data.get("baseline_metrics")
    if df is None:
        return "<p>No baseline metrics available.</p>"

    # Metric explanation tooltips shown above the table
    html = '<div style="margin-bottom:12px;">'
    for col in df.columns:
        key = col.lower().replace("-", "_").replace(" ", "_")
        if key in METRIC_EXPLANATIONS:
            html += (
                f'<details style="margin-bottom:6px;">'
                f'<summary style="cursor:pointer;color:#58a6ff;font-size:0.88rem;">'
                f'What is <em>{col.replace("_", " ").title()}</em>?</summary>'
                f'<p style="font-size:0.84rem;color:#8b949e;padding:4px 0 0 12px;">'
                f'{METRIC_EXPLANATIONS[key]}</p></details>'
            )
    html += '</div>'

    # Table
    html += _df_to_dark_table(df)
    return html


def _build_multiview_section(data: Dict[str, Any]) -> str:
    """Render ARI and NMI matrices side by side with explanations."""
    ari = data.get("ari_matrix")
    nmi = data.get("nmi_matrix")

    if ari is None and nmi is None:
        return "<p>No multi-view comparison data available.</p>"

    html = ""
    if ari is not None:
        html += '<h3>Adjusted Rand Index (ARI) Matrix</h3>'
        html += (
            f'<div class="section-info" style="margin-bottom:10px;">'
            f'{METRIC_EXPLANATIONS["ari"]}</div>'
        )
        html += f'<div class="data-table-wrap">{_df_to_dark_table(ari, fmt=".3f")}</div>'

    if nmi is not None:
        html += '<h3>Normalized Mutual Information (NMI) Matrix</h3>'
        html += (
            f'<div class="section-info" style="margin-bottom:10px;">'
            f'{METRIC_EXPLANATIONS["nmi"]}</div>'
        )
        html += f'<div class="data-table-wrap">{_df_to_dark_table(nmi, fmt=".3f")}</div>'

    return html


def _build_cluster_summary_section(data: Dict[str, Any]) -> str:
    df = data.get("cluster_summary")
    if df is None:
        return "<p>No cluster summary available.</p>"

    html = (
        f'<div style="margin-bottom:10px;">'
        f'<details><summary style="cursor:pointer;color:#58a6ff;font-size:0.88rem;">'
        f"What is Moran's I?</summary>"
        f'<p style="font-size:0.84rem;color:#8b949e;padding:4px 0 0 12px;">'
        f'{METRIC_EXPLANATIONS["morans_i"]}</p></details></div>'
    )
    html += _df_to_dark_table(df, fmt=".4f")
    return html


def _build_sensitivity_section(data: Dict[str, Any]) -> str:
    """Build sensitivity analysis summary if available."""
    sens = data.get("sensitivity")
    if sens is None:
        return ""

    import numpy as np

    combos = sens.get("combinations", [])
    ari_scores = np.array(sens.get("ari_scores", []))
    sil_scores = np.array(sens.get("silhouette_scores", []))
    n_clusters = np.array(sens.get("n_clusters", []))

    html = (
        '<h2>Sensitivity Analysis</h2>'
        f'<div class="section-info">{SECTION_EXPLANATIONS["sensitivity"]}</div>'
    )

    # Summary cards
    robust_count = int(np.sum(ari_scores >= 0.8)) if ari_scores.size else 0
    total_count = len(combos)

    html += '<div class="metrics-grid">'
    html += _metric_card(
        "Combinations tested", str(total_count), _quality_class(1)
    )
    html += _metric_card(
        "Robust (ARI &ge; 0.8)", f"{robust_count}/{total_count}",
        "good" if robust_count >= total_count * 0.7 else "warn"
    )
    if ari_scores.size:
        html += _metric_card(
            "Mean ARI", f"{float(np.mean(ari_scores)):.3f}",
            _quality_class(float(np.mean(ari_scores)), thresholds=(0.6, 0.8))
        )
    if sil_scores.size:
        html += _metric_card(
            "Best Silhouette", f"{float(np.max(sil_scores)):.3f}",
            _quality_class(float(np.max(sil_scores)), thresholds=(0.2, 0.4))
        )
    if n_clusters.size:
        unique_k = np.unique(n_clusters[n_clusters > 0])
        html += _metric_card(
            "Cluster counts seen",
            ", ".join(str(int(k)) for k in unique_k),
            _quality_class(1)
        )
    html += '</div>'

    # Top-5 weight combinations table
    if ari_scores.size and sil_scores.size:
        order = np.argsort(-sil_scores)[:5]
        html += '<h3>Top 5 Weight Combinations (by Silhouette)</h3>'
        html += (
            '<table><thead><tr>'
            '<th>&alpha;</th><th>&beta;</th><th>&gamma;</th>'
            '<th>Silhouette</th><th>ARI vs. baseline</th><th># Clusters</th>'
            '</tr></thead><tbody>'
        )
        for i in order:
            a, b, g = combos[i]
            html += (
                f'<tr><td>{a:.1f}</td><td>{b:.1f}</td><td>{g:.1f}</td>'
                f'<td>{sil_scores[i]:.3f}</td><td>{ari_scores[i]:.3f}</td>'
                f'<td>{int(n_clusters[i])}</td></tr>'
            )
        html += '</tbody></table>'

    return html


def _build_array_summaries_section(data: Dict[str, Any]) -> str:
    """Compact table of all saved numpy arrays with summary stats."""
    arrays = data.get("array_summaries", {})
    if not arrays:
        return ""

    html = (
        '<h2>Saved Data Arrays</h2>'
        '<div class="section-info">'
        "Summary statistics for all numpy arrays stored during the pipeline "
        "run. These files contain raw similarity matrices, cluster label "
        "vectors, selected gene indices, and intermediate results."
        '</div>'
        '<table><thead><tr>'
        '<th>Name</th><th>Shape</th><th>Dtype</th>'
        '<th>Min</th><th>Max</th><th>Mean</th>'
        '</tr></thead><tbody>'
    )
    for name, info in sorted(arrays.items()):
        html += (
            f'<tr>'
            f'<td><code>{name}</code></td>'
            f'<td>{info.get("shape", "?")}</td>'
            f'<td>{info.get("dtype", "?")}</td>'
            f'<td>{info.get("min", "&mdash;")}</td>'
            f'<td>{info.get("max", "&mdash;")}</td>'
            f'<td>{info.get("mean", "&mdash;")}</td>'
            f'</tr>'
        )
    html += '</tbody></table>'
    return html


def _build_plots_section(data: Dict[str, Any]) -> str:
    """Build the visualizations section with explanations per plot."""
    plots = data.get("plots", [])
    if not plots:
        return "<p>No visualizations available.</p>"

    # Group plots by notebook section for sub-headings
    section_labels = [
        ("nb01_", "Notebook 01 &mdash; Data Exploration"),
        ("nb02_", "Notebook 02 &mdash; Baseline Clustering"),
        ("nb03_", "Notebook 03 &mdash; Spatial Weighted Similarity"),
        ("nb04_", "Notebook 04 &mdash; Multi-View Clustering"),
        ("nb05_", "Notebook 05 &mdash; Final Plots"),
        # NB05 publication plots (without nb05_ prefix) grouped here too
    ]

    html = ""
    current_section = None

    for plot in plots:
        name = plot["name"]

        # Determine section heading
        section = None
        for prefix, label in section_labels:
            if name.startswith(prefix):
                section = label
                break
        # NB05 publication plots
        if section is None and name in (
            "all_similarity_matrices", "ari_nmi_matrices",
            "spatial_coherence_bar", "weighted_similarity_matrix"
        ):
            section = "Notebook 05 &mdash; Final Plots"
        # NB06 sensitivity plots
        if section is None and name.startswith(("resolution", "weight_sensitivity", "pca")):
            section = "Notebook 06 &mdash; Sensitivity Analysis"

        if section and section != current_section:
            html += f'<h3 style="margin-top:32px;color:#58a6ff;">{section}</h3>'
            current_section = section

        # Look up title and caption: exact match first, then prefix pattern
        info = PLOT_EXPLANATIONS.get(name)
        if info is None:
            info = _match_pattern_caption(name)
        title = info.get("title", name.replace("_", " ").title()) if info else name.replace("_", " ").title()
        caption = info.get("caption", "") if info else ""

        with open(plot["path"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        html += (
            f'<div class="plot-section">'
            f'<h3>{title}</h3>'
            f'<div class="plot-explanation">{caption}</div>'
            f'<img src="data:image/png;base64,{b64}" alt="{name}">'
            f'</div>'
        )

    return html


def _match_pattern_caption(name: str) -> Optional[Dict[str, str]]:
    """Match a plot name against PLOT_PATTERN_CAPTIONS prefixes."""
    for prefix, info in PLOT_PATTERN_CAPTIONS:
        if name.startswith(prefix):
            # Extract the dynamic suffix for a more specific title
            suffix = name[len(prefix):]
            title = f"{info['title_prefix']} &mdash; {suffix.replace('_', ' ')}"
            return {"title": title, "caption": info["caption"]}
    return None


def _build_detailed_results(data: Dict[str, Any]) -> str:
    return (
        "<p>All raw data files (numpy arrays, CSV tables, log files) are "
        f'stored in the session directory: <code>{data["session_id"]}</code></p>'
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_dark_table(df, fmt: str = ".4f") -> str:
    """Convert a pandas DataFrame to an HTML table styled for the dark theme."""

    def _fmt(v):
        try:
            return f"{float(v):{fmt}}"
        except (ValueError, TypeError):
            return str(v)

    html = "<table><thead><tr><th></th>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    for idx, row in df.iterrows():
        html += f"<tr><td><strong>{idx}</strong></td>"
        for val in row:
            html += f"<td>{_fmt(val)}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html


def _metric_card(label: str, value: str, quality: str = "") -> str:
    cls = f" {quality}" if quality else ""
    return (
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value{cls}">{value}</div>'
        f'</div>'
    )


def _quality_class(
    value: float, thresholds: tuple = (0.5, 0.8)
) -> str:
    """Return 'good', 'warn', or 'bad' CSS class based on thresholds."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if v >= thresholds[1]:
        return "good"
    elif v >= thresholds[0]:
        return "warn"
    else:
        return "bad"


def _generate_pdf_report(
    session: SessionManager, html_content: Optional[str] = None
) -> Path:
    """Generate PDF report from HTML."""
    from weasyprint import HTML

    pdf_path = session.run_dir / "report.pdf"

    if html_content is None:
        html_path = session.run_dir / "report.html"
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    else:
        HTML(string=html_content).write_pdf(str(pdf_path))

    return pdf_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate analysis report")
    parser.add_argument(
        "--session", type=str,
        help="Session ID (default: current session)"
    )
    parser.add_argument(
        "--format", type=str, choices=["html", "pdf", "both"],
        default="both", help="Output format"
    )
    parser.add_argument(
        "--output", type=Path, help="Custom output path"
    )

    args = parser.parse_args()

    try:
        generate_report(
            session_id=args.session,
            output_format=args.format,
            output_path=args.output
        )
    except Exception as e:
        print(f"Report generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
