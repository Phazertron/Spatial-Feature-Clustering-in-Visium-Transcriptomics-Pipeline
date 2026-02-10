from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseSimilarity(ABC):
    """
    Abstract base class for all similarity methods.
    Ensures a consistent API across different similarity implementations.
    """

    @abstractmethod
    def compute(self, X: np.ndarray) -> np.ndarray:
        """
        Computes a similarity matrix for the given feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_items × n_features)

        Returns
        -------
        np.ndarray
            Similarity matrix (n_items × n_items)
        """
        pass
