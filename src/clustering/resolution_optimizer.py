from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Literal
import warnings

from src.clustering.gene_clustering import GeneClustering
from src.evaluation.metrics import ClusteringEvaluator


class ResolutionOptimizer:
    """
    Automatically finds the optimal resolution parameter for clustering
    by maximizing Silhouette score or Moran's I spatial autocorrelation.

    This addresses the issue of hardcoded resolution=1.0, which may not
    be optimal for all datasets or similarity views.
    """

    def __init__(
        self,
        method: Literal["leiden", "louvain", "spectral"] = "louvain",
        metric: Literal["silhouette", "morans_i", "both"] = "silhouette",
        random_state: int = 0,
    ):
        """
        Parameters
        ----------
        method : {"leiden", "louvain", "spectral"}
            Clustering algorithm to use.
        metric : {"silhouette", "morans_i", "both"}
            Optimization metric:
            - "silhouette": Maximize silhouette score (faster)
            - "morans_i": Maximize average Moran's I (spatial coherence)
            - "both": Use silhouette for speed, validate with Moran's I
        random_state : int
            Random seed for reproducibility.
        """
        self.method = method
        self.metric = metric
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Grid Search
    # ------------------------------------------------------------------

    def grid_search(
        self,
        similarity_matrix: np.ndarray,
        resolution_range: Optional[List[float]] = None,
        adata = None,
        gene_indices: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Performs grid search over resolution values to find the optimal one.

        Parameters
        ----------
        similarity_matrix : np.ndarray
            Precomputed similarity matrix (n_genes × n_genes).
        resolution_range : list of float, optional
            Resolutions to test. Default: [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        adata : AnnData, optional
            Required if metric includes "morans_i".
        gene_indices : np.ndarray, optional
            Gene indices for Moran's I computation.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        dict
            {
                "optimal_resolution": float,
                "optimal_score": float,
                "results": list of dicts with all tested resolutions,
                "optimal_labels": np.ndarray,
            }
        """
        if resolution_range is None:
            resolution_range = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]

        if self.metric in ["morans_i", "both"] and (adata is None or gene_indices is None):
            raise ValueError("adata and gene_indices required for Moran's I computation")

        results = []
        best_score = -np.inf
        best_resolution = resolution_range[0]
        best_labels = None

        for res in resolution_range:
            if verbose:
                print(f"Testing resolution={res:.2f}...", end=" ")

            # Cluster with this resolution
            clusterer = GeneClustering(
                method=self.method,
                resolution=res,
                random_state=self.random_state,
            )
            labels = clusterer.cluster(similarity_matrix)

            # Check if we got at least 2 clusters
            n_clusters = len(np.unique(labels))
            if n_clusters < 2:
                if verbose:
                    print(f"Only {n_clusters} cluster(s) found, skipping.")
                results.append({
                    "resolution": res,
                    "n_clusters": n_clusters,
                    "silhouette": np.nan,
                    "morans_i": np.nan,
                    "score": -np.inf,
                })
                continue

            # Compute metrics
            silhouette = np.nan
            morans_i = np.nan

            if self.metric in ["silhouette", "both"]:
                silhouette = ClusteringEvaluator.compute_basic_metrics(
                    similarity_matrix, labels
                )["silhouette"]

            if self.metric in ["morans_i", "both"]:
                coherence_dict = ClusteringEvaluator.compute_cluster_spatial_coherence(
                    adata, labels
                )
                morans_i = np.mean(list(coherence_dict.values()))

            # Determine score for optimization
            if self.metric == "silhouette":
                score = silhouette
            elif self.metric == "morans_i":
                score = morans_i
            else:  # both
                # Use silhouette as primary, Moran's I as secondary validation
                score = silhouette

            results.append({
                "resolution": res,
                "n_clusters": n_clusters,
                "silhouette": silhouette,
                "morans_i": morans_i,
                "score": score,
            })

            # Update best
            if score > best_score:
                best_score = score
                best_resolution = res
                best_labels = labels

            if verbose:
                print(f"n_clusters={n_clusters}, silhouette={silhouette:.3f}, "
                      f"morans_i={morans_i:.3f}")

        if verbose:
            print(f"\nOptimal resolution: {best_resolution:.2f} "
                  f"(score={best_score:.3f})")

        return {
            "optimal_resolution": best_resolution,
            "optimal_score": best_score,
            "results": results,
            "optimal_labels": best_labels,
        }

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def optimize(
        self,
        similarity_matrix: np.ndarray,
        **kwargs
    ) -> float:
        """
        Convenience method that returns only the optimal resolution value.

        Parameters
        ----------
        similarity_matrix : np.ndarray
            Similarity matrix.
        **kwargs
            Additional arguments passed to grid_search.

        Returns
        -------
        float
            Optimal resolution value.
        """
        result = self.grid_search(similarity_matrix, **kwargs)
        return result["optimal_resolution"]
