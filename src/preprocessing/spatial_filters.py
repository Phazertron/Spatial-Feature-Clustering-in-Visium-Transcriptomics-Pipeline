from __future__ import annotations
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from typing import Optional


class SpatialFilterBank:
    """
    Encapsulates all spatial filtering operations used in the project.
    This replaces the original filt.py, removing global variables and
    making the system modular and object-oriented.
    """

    # Default parameters
    DEFAULT_MEAN_FILTER_IT = 6
    DEFAULT_EXP_FILTER_IT = 1

    # Hexagonal Visium neighborhood (7 neighbors)
    NEIGHBOR_OFFSETS = np.array([
        (0, 0),
        (0, -2),
        (0, +2),
        (-1, -1),
        (-1, +1),
        (+1, -1),
        (+1, +1)
    ], dtype=int)

    def __init__(self, adata):
        """
        Initializes the spatial filter bank by computing the neighborhood
        graph for the Visium grid.

        Parameters
        ----------
        adata : AnnData
            The loaded Visium dataset.
        """
        self.adata = adata
        self._initialize_neighbors()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_neighbors(self):
        obs = self.adata.obs
        # 1. Map (row, col) tuple -> integer index safely
        coord_map = { (int(r), int(c)): i for i, (r, c) in enumerate(zip(obs.array_row, obs.array_col)) }
        
        n_spots = len(obs)
        self.ITNI = np.full((n_spots, 7), -1, dtype=int)

        for i, (r, c) in enumerate(zip(obs.array_row, obs.array_col)):
            r, c = int(r), int(c)
            # Apply offsets to current r, c
            for neighbor_pos, offset in enumerate(self.NEIGHBOR_OFFSETS):
                target = (r + offset[0], c + offset[1])
                self.ITNI[i, neighbor_pos] = coord_map.get(target, -1)

        # Re-calculate masks
        self.VNI = (self.ITNI != -1).astype(int)
        self.NN = self.VNI.sum(axis=1)
        self.NM = (self.VNI.T / np.maximum(self.NN, 1)).T
        self.ITSNI = self.ITNI[:, 1:]

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def max_filter(self, gene_data: np.ndarray) -> np.ndarray:
        """
        Applies a max filter over the neighborhood.
        """
        return np.max(gene_data[self.ITNI] * self.VNI, axis=1)

    def mean_filter(self, gene_data: np.ndarray, iterations: int) -> np.ndarray:
        """
        Applies the optimized mean filter iteratively.
        """
        filtered = gene_data.copy()
        for _ in range(iterations):
            filtered = (filtered[self.ITNI] * self.NM).sum(axis=1)
        return filtered

    def expansion_filter(self, gene_data: np.ndarray, iterations: int) -> np.ndarray:
        """
        Applies the expansion filter iteratively.
        """
        filtered = gene_data.copy()
        for _ in range(iterations):
            active = filtered[self.ITSNI].sum(axis=1)
            filtered = (active >= 2).astype(int) * filtered + (active > 2).astype(int) * (1 - filtered)
        return filtered

    # ------------------------------------------------------------------
    # Gaussian Mixture Transform
    # ------------------------------------------------------------------

    def transform_mog(
        self,
        gene_data: np.ndarray,
        mean_filter_it: int = DEFAULT_MEAN_FILTER_IT,
        exp_filter_it: int = DEFAULT_EXP_FILTER_IT,
        threshold: float = 0.3  # New parameter: lower means more connectivity
    ) -> np.ndarray:
        """
        Smooths → fits 2 Gaussians → soft binarizes → expands.
        """
        # 1. Smooth
        smoothed = self.mean_filter(gene_data, mean_filter_it)

        # 2. Fit GMM
        gm = GaussianMixture(n_components=2, random_state=0).fit(smoothed.reshape(-1, 1))
        
        # 3. Get probabilities instead of hard labels
        probs = gm.predict_proba(smoothed.reshape(-1, 1))

        # Ensure we are looking at the probability of the "higher expression" component
        m1, m2 = gm.means_.flatten()
        high_comp_idx = 1 if m2 > m1 else 0
        
        # 4. Apply a custom threshold (e.g., 0.3 instead of 0.5)
        labels = (probs[:, high_comp_idx] >= threshold).astype(int)

        # 5. Expand (using your updated 'active >= 2' logic from suggestion A)
        expanded = self.expansion_filter(labels, exp_filter_it)
        return expanded

    # ------------------------------------------------------------------
    # Batch transform
    # ------------------------------------------------------------------

    def transform_genes(
        self,
        gene_indices: np.ndarray,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Applies the MoG transform to multiple genes.

        Parameters
        ----------
        gene_indices : np.ndarray
            Array of gene IDs.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        np.ndarray
            Transformed gene matrix (genes × spots).
        """
        array_data = self.adata.X.toarray().T

        n_genes = len(gene_indices)
        n_spots = array_data.shape[1]

        transformed = np.zeros((n_genes, n_spots), dtype=int)

        for i, gene_id in enumerate(gene_indices):
            if verbose and i % 100 == 0:
                print(f"{i}/{n_genes} processed")

            gene_data = array_data[gene_id]
            transformed[i] = self.transform_mog(gene_data)

        return transformed

    # ------------------------------------------------------------------
    # Dimensionality Reduction
    # ------------------------------------------------------------------

    @staticmethod
    def apply_pca(
        X: np.ndarray,
        n_components: int = 50,
        random_state: int = 0,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Applies PCA for dimensionality reduction.

        This addresses the warning about high dimensionality during
        neighbor computation in Scanpy. Explicit PCA before neighbor
        computation speeds up processing and reduces noise.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples × n_features).
        n_components : int
            Number of principal components to retain.
        random_state : int
            Random seed.
        verbose : bool
            Whether to print explained variance.

        Returns
        -------
        np.ndarray
            Reduced feature matrix (n_samples × n_components).
        """
        n_samples, n_features = X.shape
        n_components = min(n_components, n_samples, n_features)

        pca = PCA(n_components=n_components, random_state=random_state)
        X_reduced = pca.fit_transform(X)

        if verbose:
            total_variance = np.sum(pca.explained_variance_ratio_)
            print(f"PCA: {n_components} components explain "
                  f"{total_variance*100:.2f}% of variance")

        return X_reduced
