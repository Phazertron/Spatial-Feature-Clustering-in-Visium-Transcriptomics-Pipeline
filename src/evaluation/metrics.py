from __future__ import annotations
import numpy as np
from typing import Optional, Dict

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

import libpysal
import esda


class ClusteringEvaluator:
    """
    Computes clustering quality metrics and spatial autocorrelation metrics.
    """

    # ------------------------------------------------------------------
    # Basic clustering metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_basic_metrics(
        X: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes silhouette, Calinski-Harabasz, and Davies-Bouldin scores.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (genes × features).
        labels : np.ndarray
            Cluster labels.

        Returns
        -------
        dict
            Dictionary of metrics.
        """
        metrics = {}

        if len(np.unique(labels)) > 1:
            metrics["silhouette"] = silhouette_score(X, labels)
            metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
            metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
        else:
            metrics["silhouette"] = np.nan
            metrics["calinski_harabasz"] = np.nan
            metrics["davies_bouldin"] = np.nan

        return metrics

    # ------------------------------------------------------------------
    # Agreement metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compare_clusterings(
        labels_a: np.ndarray,
        labels_b: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes ARI and NMI between two clusterings.

        Parameters
        ----------
        labels_a : np.ndarray
        labels_b : np.ndarray

        Returns
        -------
        dict
            Dictionary with ARI and NMI.
        """
        return {
            "ARI": adjusted_rand_score(labels_a, labels_b),
            "NMI": normalized_mutual_info_score(labels_a, labels_b),
        }

    # ------------------------------------------------------------------
    # Spatial metrics
    # ------------------------------------------------------------------

    @staticmethod
    def morans_I(values: np.ndarray, weights) -> float:
        """
        Computes Moran's I for a vector of spatial values.

        Parameters
        ----------
        values : np.ndarray
            Values per spot.
        weights : libpysal.weights.W
            Spatial weights matrix.

        Returns
        -------
        float
            Moran's I statistic.
        """
        mi = esda.moran.Moran(values, weights)
        return mi.I

    @staticmethod
    def compute_cluster_spatial_coherence(
        adata,
        labels: np.ndarray,
        use_pca: bool = False,
        n_components: int = 50
    ) -> Dict[int, float]:
        """
        Computes Moran's I for each gene cluster.

        Parameters
        ----------
        adata : AnnData
            Dataset with spatial coordinates.
        labels : np.ndarray
            Cluster labels for genes.
        use_pca : bool
            Whether to apply PCA before computing neighbors (recommended
            for high-dimensional data to avoid warnings and speed up computation).
        n_components : int
            Number of PCA components if use_pca=True.

        Returns
        -------
        dict
            Mapping cluster_id → average Moran's I.
        """
        # Build spatial weights from Visium grid
        coords = adata.obsm["spatial"]

        # Apply PCA to coords if requested (for high-dimensional spot data)
        if use_pca and coords.shape[1] > n_components:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(n_components, coords.shape[0], coords.shape[1]))
            coords = pca.fit_transform(coords)

        weights = libpysal.weights.KNN.from_array(coords, k=6)

        cluster_ids = np.unique(labels)
        results = {}

        array_data = adata.X.toarray().T

        for cid in cluster_ids:
            gene_indices = np.where(labels == cid)[0]

            if len(gene_indices) == 0:
                results[cid] = np.nan
                continue

            morans_values = []

            for gene_id in gene_indices:
                gene_data = array_data[gene_id]
                mi = ClusteringEvaluator.morans_I(gene_data, weights)
                morans_values.append(mi)

            results[cid] = float(np.mean(morans_values))

        return results
