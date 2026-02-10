from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from itertools import product
import warnings

from src.similarity.spatial_weighted_similarity import SpatialWeightedSimilarity
from src.clustering.gene_clustering import GeneClustering
from src.evaluation.metrics import ClusteringEvaluator


class WeightSensitivityAnalyzer:
    """
    Performs sensitivity analysis on the spatial weighted similarity weights
    (alpha, beta, gamma) to identify robust weight combinations and justify
    default choices scientifically.

    This addresses the issue of arbitrary default weights (0.5, 0.3, 0.2)
    by systematically testing different combinations.
    """

    def __init__(
        self,
        dataset,
        clustering_method: str = "louvain",
        resolution: float = 1.0,
        random_state: int = 0,
    ):
        """
        Parameters
        ----------
        dataset : SpatialDataset
            Loaded spatial dataset.
        clustering_method : str
            Clustering algorithm to use.
        resolution : float
            Resolution parameter for clustering.
        random_state : int
            Random seed.
        """
        self.dataset = dataset
        self.clustering_method = clustering_method
        self.resolution = resolution
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Weight combination generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_weight_combinations(
        values: List[float] = None,
        tolerance: float = 1e-6
    ) -> List[Tuple[float, float, float]]:
        """
        Generates all valid (alpha, beta, gamma) combinations that sum to 1.0.

        Parameters
        ----------
        values : list of float, optional
            Weight values to test. Default: [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        tolerance : float
            Tolerance for sum constraint.

        Returns
        -------
        list of tuple
            List of (alpha, beta, gamma) tuples.
        """
        if values is None:
            values = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

        valid_combinations = []

        for alpha, beta, gamma in product(values, repeat=3):
            if abs(alpha + beta + gamma - 1.0) < tolerance:
                valid_combinations.append((alpha, beta, gamma))

        return valid_combinations

    # ------------------------------------------------------------------
    # Sensitivity Analysis
    # ------------------------------------------------------------------

    def analyze_sensitivity(
        self,
        gene_indices: np.ndarray,
        weight_combinations: List[Tuple[float, float, float]] = None,
        baseline_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        verbose: bool = True,
    ) -> Dict:
        """
        Performs sensitivity analysis by testing different weight combinations.

        For each combination:
        1. Compute weighted similarity matrix
        2. Cluster genes
        3. Compare to baseline clustering (ARI, NMI)
        4. Compute Silhouette score on the raw expression matrix

        The Silhouette score is always evaluated on the **raw expression**
        feature space so that values are comparable across weight
        combinations.  Using each combination's similarity matrix would
        bias scores toward representations with simpler geometry (e.g.
        pure MoG binarization), making cross-combination comparison
        unreliable.

        Parameters
        ----------
        gene_indices : np.ndarray
            Gene indices to cluster.
        weight_combinations : list of tuple, optional
            List of (alpha, beta, gamma) to test.
        baseline_weights : tuple
            Baseline weights for comparison.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        dict
            {
                "combinations": list of tuples,
                "ari_scores": np.ndarray,
                "nmi_scores": np.ndarray,
                "silhouette_scores": np.ndarray,
                "n_clusters": np.ndarray,
                "baseline_labels": np.ndarray,
            }
        """
        if weight_combinations is None:
            weight_combinations = self.generate_weight_combinations()

        if verbose:
            print(f"Testing {len(weight_combinations)} weight combinations...")

        # Build a shared feature matrix (raw expression) for Silhouette
        # so that scores are comparable across different similarity spaces.
        X_raw = self.dataset.adata.X.toarray().T[gene_indices]

        # Compute baseline clustering
        baseline_sws = SpatialWeightedSimilarity(
            self.dataset,
            alpha=baseline_weights[0],
            beta=baseline_weights[1],
            gamma=baseline_weights[2],
        )
        baseline_similarity = baseline_sws.compute_similarity_matrix(gene_indices)

        baseline_clusterer = GeneClustering(
            method=self.clustering_method,
            resolution=self.resolution,
            random_state=self.random_state,
        )
        baseline_labels = baseline_clusterer.cluster(baseline_similarity)

        # Storage for results
        ari_scores = []
        nmi_scores = []
        silhouette_scores = []
        n_clusters_list = []

        # Test each combination
        for i, (alpha, beta, gamma) in enumerate(weight_combinations):
            if verbose and i % 10 == 0:
                print(f"Progress: {i}/{len(weight_combinations)}")

            # Compute weighted similarity
            sws = SpatialWeightedSimilarity(
                self.dataset,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            similarity = sws.compute_similarity_matrix(gene_indices)

            # Cluster
            clusterer = GeneClustering(
                method=self.clustering_method,
                resolution=self.resolution,
                random_state=self.random_state,
            )
            labels = clusterer.cluster(similarity)

            # Compute metrics
            n_clusters = len(np.unique(labels))
            n_clusters_list.append(n_clusters)

            # ARI/NMI vs baseline
            comparison = ClusteringEvaluator.compare_clusterings(
                baseline_labels, labels
            )
            ari_scores.append(comparison["ARI"])
            nmi_scores.append(comparison["NMI"])

            # Silhouette on shared raw-expression space so that scores
            # are comparable across weight combinations.
            if n_clusters > 1:
                silhouette = ClusteringEvaluator.compute_basic_metrics(
                    X_raw, labels
                )["silhouette"]
            else:
                silhouette = np.nan

            silhouette_scores.append(silhouette)

        if verbose:
            print(f"Sensitivity analysis complete!")

        return {
            "combinations": weight_combinations,
            "ari_scores": np.array(ari_scores),
            "nmi_scores": np.array(nmi_scores),
            "silhouette_scores": np.array(silhouette_scores),
            "n_clusters": np.array(n_clusters_list),
            "baseline_labels": baseline_labels,
            "baseline_weights": baseline_weights,
        }

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def identify_robust_weights(
        self,
        results: Dict,
        ari_threshold: float = 0.8,
    ) -> List[Tuple[float, float, float]]:
        """
        Identifies weight combinations that produce stable clusterings
        (high ARI with baseline).

        Parameters
        ----------
        results : dict
            Output from analyze_sensitivity.
        ari_threshold : float
            Minimum ARI to consider robust.

        Returns
        -------
        list of tuple
            Robust weight combinations.
        """
        robust_mask = results["ari_scores"] >= ari_threshold
        robust_combinations = [
            results["combinations"][i]
            for i in range(len(results["combinations"]))
            if robust_mask[i]
        ]

        return robust_combinations

    def find_optimal_weights(
        self,
        results: Dict,
        metric: str = "silhouette"
    ) -> Tuple[float, float, float]:
        """
        Finds the weight combination that maximizes a given metric.

        Parameters
        ----------
        results : dict
            Output from analyze_sensitivity.
        metric : str
            Metric to optimize ("silhouette", "ari", "nmi").

        Returns
        -------
        tuple
            Optimal (alpha, beta, gamma).
        """
        if metric == "silhouette":
            scores = results["silhouette_scores"]
        elif metric == "ari":
            scores = results["ari_scores"]
        elif metric == "nmi":
            scores = results["nmi_scores"]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Handle NaN values
        valid_mask = ~np.isnan(scores)
        if not np.any(valid_mask):
            warnings.warn("No valid scores found, returning baseline weights")
            return results["baseline_weights"]

        valid_scores = scores[valid_mask]
        valid_combinations = [
            results["combinations"][i]
            for i in range(len(results["combinations"]))
            if valid_mask[i]
        ]

        optimal_idx = np.argmax(valid_scores)
        return valid_combinations[optimal_idx]
