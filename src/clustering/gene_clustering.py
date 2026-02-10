from __future__ import annotations
import numpy as np
from typing import Literal, Optional

import networkx as nx
from sklearn.cluster import SpectralClustering

try:
    import igraph as ig
    import leidenalg
    _HAS_LEIDEN = True
except ImportError:
    _HAS_LEIDEN = False


class GeneClustering:
    """
    Clusters genes based on a precomputed similarity matrix.
    Supports:
    - Leiden (if available)
    - Louvain (via NetworkX)
    - Spectral clustering (sklearn)
    """

    def __init__(
        self,
        method: Literal["leiden", "louvain", "spectral"] = "louvain",
        n_clusters: Optional[int] = None,
        resolution: float = 1.0,
        random_state: int = 0,
    ):
        """
        Parameters
        ----------
        method : {"leiden", "louvain", "spectral"}
            Clustering algorithm to use.
        n_clusters : int, optional
            Number of clusters (used for spectral).
        resolution : float
            Resolution parameter for Leiden/Louvain.
        random_state : int
            Random seed.
        """
        self.method = method
        self.n_clusters = n_clusters
        self.resolution = resolution
        self.random_state = random_state

        if self.method == "leiden" and not _HAS_LEIDEN:
            raise ImportError("Leiden not available. Install `python-igraph` and `leidenalg`.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Clusters genes based on the given similarity matrix.

        Parameters
        ----------
        similarity_matrix : np.ndarray
            Symmetric similarity matrix (n_genes × n_genes).

        Returns
        -------
        np.ndarray
            Cluster labels for each gene.
        """
        if self.method == "spectral":
            return self._cluster_spectral(similarity_matrix)
        elif self.method == "leiden":
            return self._cluster_leiden(similarity_matrix)
        elif self.method == "louvain":
            return self._cluster_louvain(similarity_matrix)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _similarity_to_graph(self, S: np.ndarray) -> nx.Graph:
        """
        Converts a similarity matrix into a weighted graph.
        """
        np.fill_diagonal(S, 0.0)
        G = nx.from_numpy_array(S)
        return G

    def _cluster_louvain(self, S: np.ndarray) -> np.ndarray:
        """
        Louvain clustering via NetworkX community.
        """
        import networkx.algorithms.community as nx_comm

        G = self._similarity_to_graph(S)
        communities = nx_comm.louvain_communities(
            G,
            weight="weight",
            resolution=self.resolution,
            seed=self.random_state,
        )

        labels = np.zeros(S.shape[0], dtype=int)
        for cid, comm in enumerate(communities):
            for node in comm:
                labels[node] = cid

        return labels

    def _cluster_leiden(self, S: np.ndarray) -> np.ndarray:
        """
        Leiden clustering via igraph + leidenalg.
        """
        np.fill_diagonal(S, 0.0)
        sources, targets = np.where(S > 0)
        weights = S[sources, targets]

        g = ig.Graph(directed=False)
        g.add_vertices(S.shape[0])
        edges = list(zip(sources.tolist(), targets.tolist()))
        g.add_edges(edges)
        g.es["weight"] = weights.tolist()

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=self.resolution,
            seed=self.random_state,
        )

        labels = np.zeros(S.shape[0], dtype=int)
        for cid, comm in enumerate(partition):
            for node in comm:
                labels[node] = cid

        return labels

    def _cluster_spectral(self, S: np.ndarray) -> np.ndarray:
        """
        Spectral clustering using sklearn.
        """
        if self.n_clusters is None:
            raise ValueError("n_clusters must be set for spectral clustering.")

        # Convert similarity to affinity
        np.fill_diagonal(S, 1.0)

        model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity="precomputed",
            random_state=self.random_state,
        )

        labels = model.fit_predict(S)
        return labels
