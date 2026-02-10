from __future__ import annotations
import numpy as np
from typing import List

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

from src.data.data_loader import SpatialDataset
from src.preprocessing.spatial_filters import SpatialFilterBank


class SpatialWeightedSimilarity:
    """
    Computes a combined similarity between genes using:
    - raw expression similarity
    - spatially filtered similarity
    - MoG-transformed similarity
    """

    def __init__(
        self,
        dataset: SpatialDataset,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
    ):
        """
        Parameters
        ----------
        dataset : SpatialDataset
            Loaded dataset with filter_bank initialized.
        alpha, beta, gamma : float
            Weights for expression, spatial, and MoG similarities.
        """
        self.dataset = dataset
        self.filter_bank: SpatialFilterBank = dataset.filter_bank

        if not np.isclose(alpha + beta + gamma, 1.0):
            raise ValueError("alpha + beta + gamma must sum to 1")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Core similarity components
    # ------------------------------------------------------------------

    def _expression_similarity(self, X: np.ndarray) -> np.ndarray:
        return cosine_similarity(X)

    def _spatial_similarity(self, X: np.ndarray) -> np.ndarray:
        return cosine_similarity(X)

    def _mog_similarity(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        S = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                s = jaccard_score(X[i], X[j])
                S[i, j] = S[j, i] = s

        return S

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_similarity_matrix(self, gene_indices: np.ndarray) -> np.ndarray:
        """
        Computes the full weighted similarity matrix for a set of genes.

        Parameters
        ----------
        gene_indices : np.ndarray
            Array of gene IDs.

        Returns
        -------
        np.ndarray
            Weighted similarity matrix (n_genes × n_genes).
        """
        array_data = self.dataset.adata.X.toarray().T
        raw_expr = array_data[gene_indices]

        # Spatially filtered
        spatial_filtered = np.array([
            self.filter_bank.mean_filter(raw_expr[i], 6)
            for i in range(len(gene_indices))
        ])

        # MoG-transformed
        mog_transformed = np.array([
            self.filter_bank.transform_mog(raw_expr[i])
            for i in range(len(gene_indices))
        ])

        # Compute components
        S_expr = self._expression_similarity(raw_expr)
        S_spat = self._spatial_similarity(spatial_filtered)
        S_mog = self._mog_similarity(mog_transformed)

        # Weighted combination
        S = (
            self.alpha * S_expr +
            self.beta * S_spat +
            self.gamma * S_mog
        )

        return S
