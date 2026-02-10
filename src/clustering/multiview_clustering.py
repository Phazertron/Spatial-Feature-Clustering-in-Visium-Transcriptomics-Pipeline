from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.similarity.spatial_weighted_similarity import SpatialWeightedSimilarity
from src.clustering.gene_clustering import GeneClustering
from src.clustering.resolution_optimizer import ResolutionOptimizer
from src.evaluation.metrics import ClusteringEvaluator


class MultiViewClustering:
    """
    Computes multiple clusterings of the same gene set using different
    similarity views, then compares them to identify alternative structures.
    """

    def __init__(
        self,
        dataset,
        clustering_method: str = "louvain",
        resolution: float = 1.0,
        random_state: int = 0,
        save_to_adata: bool = True,
        optimize_resolution: bool = False,
        resolution_range: Optional[List[float]] = None,
        weights: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Parameters
        ----------
        dataset : SpatialDataset
            Loaded dataset.
        clustering_method : str
            "louvain", "leiden", or "spectral".
        resolution : float
            Resolution parameter for graph-based clustering (used if
            optimize_resolution=False).
        random_state : int
            Random seed.
        save_to_adata : bool
            Whether to automatically save cluster labels to adata.obs.
        optimize_resolution : bool
            Whether to automatically optimize resolution for each view.
            If True, finds optimal resolution by maximizing Silhouette score.
        resolution_range : list of float, optional
            Resolutions to test if optimize_resolution=True.
        weights : tuple of float, optional
            Custom (alpha, beta, gamma) weights for weighted similarity.
            If None, uses default (0.5, 0.3, 0.2).
        """
        self.dataset = dataset
        self.clustering_method = clustering_method
        self.resolution = resolution
        self.random_state = random_state
        self.save_to_adata = save_to_adata
        self.optimize_resolution = optimize_resolution
        self.resolution_range = resolution_range
        self.weights = weights if weights is not None else (0.5, 0.3, 0.2)
        self.optimized_resolutions = {}  # Store optimal resolutions per view

    # ------------------------------------------------------------------
    # Compute all similarity views
    # ------------------------------------------------------------------

    def compute_views(self, gene_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Computes multiple similarity matrices for the same gene set.

        Returns
        -------
        dict
            Mapping view_name → similarity_matrix
        """
        # Use custom weights if specified
        sws = SpatialWeightedSimilarity(
            self.dataset,
            alpha=self.weights[0],
            beta=self.weights[1],
            gamma=self.weights[2],
        )

        array_data = self.dataset.adata.X.toarray().T
        raw_expr = array_data[gene_indices]

        # View 1: Expression similarity only
        S_expr = sws._expression_similarity(raw_expr)

        # View 2: Spatial filtered similarity only
        spatial_filtered = np.array([
            self.dataset.filter_bank.mean_filter(raw_expr[i], 6)
            for i in range(len(gene_indices))
        ])
        S_spatial = sws._spatial_similarity(spatial_filtered)

        # View 3: MoG similarity only
        mog_transformed = np.array([
            self.dataset.filter_bank.transform_mog(raw_expr[i])
            for i in range(len(gene_indices))
        ])
        S_mog = sws._mog_similarity(mog_transformed)

        # View 4: Full weighted similarity (with custom weights)
        S_weighted = sws.compute_similarity_matrix(gene_indices)

        return {
            "expression": S_expr,
            "spatial": S_spatial,
            "mog": S_mog,
            "weighted": S_weighted,
        }

    # ------------------------------------------------------------------
    # Cluster each view
    # ------------------------------------------------------------------

    def cluster_views(
        self,
        similarity_views: Dict[str, np.ndarray],
        gene_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Clusters each similarity matrix independently.

        If optimize_resolution=True, finds optimal resolution for each view.
        Otherwise, uses the fixed resolution parameter.

        Parameters
        ----------
        similarity_views : dict
            Mapping view_name → similarity_matrix
        gene_indices : np.ndarray, optional
            Gene indices (needed for resolution optimization with Moran's I)

        Returns
        -------
        dict
            Mapping view_name → cluster_labels
        """
        results = {}

        for name, S in similarity_views.items():
            if self.optimize_resolution:
                # Optimize resolution for this view
                optimizer = ResolutionOptimizer(
                    method=self.clustering_method,
                    metric="silhouette",  # Use silhouette for speed
                    random_state=self.random_state,
                )

                opt_result = optimizer.grid_search(
                    S,
                    resolution_range=self.resolution_range,
                    verbose=True,
                )

                labels = opt_result["optimal_labels"]
                self.optimized_resolutions[name] = opt_result["optimal_resolution"]

                print(f"  View '{name}': optimal resolution = "
                      f"{opt_result['optimal_resolution']:.2f}")
            else:
                # Use fixed resolution
                clusterer = GeneClustering(
                    method=self.clustering_method,
                    resolution=self.resolution,
                    random_state=self.random_state,
                )
                labels = clusterer.cluster(S)

            results[name] = labels

        return results

    # ------------------------------------------------------------------
    # Compare clusterings
    # ------------------------------------------------------------------

    def compare_views(
        self,
        clusterings: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes ARI/NMI between all pairs of clusterings.

        Returns
        -------
        dict
            Nested dict: viewA → viewB → metrics
        """
        views = list(clusterings.keys())
        results = {}

        for i, v1 in enumerate(views):
            results[v1] = {}
            for j, v2 in enumerate(views):
                if i == j:
                    results[v1][v2] = {"ARI": 1.0, "NMI": 1.0}
                else:
                    metrics = ClusteringEvaluator.compare_clusterings(
                        clusterings[v1],
                        clusterings[v2]
                    )
                    results[v1][v2] = metrics

        return results

    # ------------------------------------------------------------------
    # Save cluster labels to AnnData
    # ------------------------------------------------------------------

    def save_cluster_labels_to_adata(
        self,
        clusterings: Dict[str, np.ndarray],
        gene_indices: np.ndarray
    ):
        """
        Saves cluster labels to adata.obs for easy access and visualization.

        For each view, creates columns in adata.obs with the average expression
        of each cluster across all spots. This enables spatial plotting of
        cluster coherence.

        Parameters
        ----------
        clusterings : dict
            Mapping view_name → cluster_labels
        gene_indices : np.ndarray
            Gene indices used in clustering
        """
        for view_name, labels in clusterings.items():
            unique_clusters = np.unique(labels)

            for cluster_id in unique_clusters:
                # Find genes in this cluster
                genes_in_cluster = gene_indices[labels == cluster_id]

                # Compute average expression across cluster
                cluster_expression = self.dataset.adata[:, genes_in_cluster].X.mean(axis=1)

                # Save to adata.obs
                col_name = f"cluster_{view_name}_{cluster_id}"
                self.dataset.adata.obs[col_name] = np.array(cluster_expression).flatten()

            # Also save the raw cluster labels for each view
            # This creates a mapping that can be used for coherence computation
            # Note: We can't directly store gene cluster labels in adata.obs
            # (since obs is for spots, not genes), but we store them in adata.uns
            if "gene_clusters" not in self.dataset.adata.uns:
                self.dataset.adata.uns["gene_clusters"] = {}

            self.dataset.adata.uns["gene_clusters"][view_name] = labels

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, gene_indices: np.ndarray):
        """
        Runs the full multi-view clustering pipeline.

        Returns
        -------
        dict
            {
                "similarities": ...,
                "clusterings": ...,
                "comparisons": ...,
                "gene_indices": ...,
                "optimized_resolutions": ... (if optimize_resolution=True)
            }
        """
        similarities = self.compute_views(gene_indices)
        clusterings = self.cluster_views(similarities, gene_indices)
        comparisons = self.compare_views(clusterings)

        # Save cluster labels to adata if requested
        if self.save_to_adata:
            self.save_cluster_labels_to_adata(clusterings, gene_indices)

        result = {
            "similarities": similarities,
            "clusterings": clusterings,
            "comparisons": comparisons,
            "gene_indices": gene_indices,
        }

        if self.optimize_resolution:
            result["optimized_resolutions"] = self.optimized_resolutions

        return result
