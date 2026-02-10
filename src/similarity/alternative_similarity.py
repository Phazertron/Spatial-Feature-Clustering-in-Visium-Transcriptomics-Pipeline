from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import mutual_info_regression

from src.similarity.base_similarity import BaseSimilarity


class PearsonSimilarity(BaseSimilarity):
    """Pearson correlation similarity."""

    def compute(self, X: np.ndarray) -> np.ndarray:
        return np.corrcoef(X)


class SpearmanSimilarity(BaseSimilarity):
    """Spearman rank correlation similarity."""

    def compute(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        S = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                rho, _ = spearmanr(X[i], X[j])
                S[i, j] = S[j, i] = rho

        return S


class CosineSimilarity(BaseSimilarity):
    """Cosine similarity."""

    def compute(self, X: np.ndarray) -> np.ndarray:
        return cosine_similarity(X)


class MutualInformationSimilarity(BaseSimilarity):
    """Mutual information similarity."""

    def compute(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        S = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                mi = mutual_info_regression(
                    X[i].reshape(-1, 1),
                    X[j]
                )[0]
                S[i, j] = S[j, i] = mi

        return S


class DistanceCorrelationSimilarity(BaseSimilarity):
    """Distance correlation similarity."""

    def compute(self, X: np.ndarray) -> np.ndarray:
        # Convert distance to similarity
        D = pairwise_distances(X, metric="euclidean")
        S = 1 / (1 + D)
        return S
