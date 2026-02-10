from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

import squidpy as sq

from src.preprocessing.spatial_filters import SpatialFilterBank


# ----------------------------------------------------------------------
# Global style configuration
# ----------------------------------------------------------------------

W = "#f8f5ff"
B = "#1e1e1e"

mpl.rcParams["text.color"] = W
mpl.rcParams["axes.labelcolor"] = W
mpl.rcParams["axes.edgecolor"] = W
mpl.rcParams["axes.facecolor"] = B
mpl.rcParams["figure.facecolor"] = B
mpl.rcParams["xtick.color"] = W
mpl.rcParams["ytick.color"] = W
mpl.rcParams["font.family"] = "monospace"

# Black–white colormap used for binarized maps
BW_COLORS = [(0.12, 0, 0), (1, 1, 1)]
BW_CMAP = LinearSegmentedColormap.from_list("black_white", BW_COLORS)

DEFAULT_SHAPE = "hex"
DEFAULT_CMAP = "bone"
DEFAULT_FIGSIZE = (5, 5)
DEFAULT_IMG_ALPHA = 0.0
DEFAULT_COLORBAR = False


class SpatialPlotter:
    """
    High-level plotting utilities for spatial transcriptomics data.
    Wraps Squidpy/Matplotlib calls into reusable methods.
    """

    def __init__(
        self,
        adata,
        filter_bank: Optional[SpatialFilterBank] = None,
        shape: str = DEFAULT_SHAPE,
        cmap: str | LinearSegmentedColormap = DEFAULT_CMAP,
        figsize: tuple[int, int] = DEFAULT_FIGSIZE,
        img_alpha: float = DEFAULT_IMG_ALPHA,
        show_colorbar: bool = DEFAULT_COLORBAR,
    ):
        """
        Parameters
        ----------
        adata : AnnData
            The loaded Visium dataset.
        filter_bank : SpatialFilterBank, optional
            The spatial filter system used for MoG transforms.
        shape : str
            Spot shape for spatial plots ("hex", "circle", etc.).
        cmap : str or Colormap
            Default colormap for continuous plots.
        figsize : tuple
            Default figure size.
        img_alpha : float
            Background image alpha.
        show_colorbar : bool
            Whether to show colorbars by default.
        """
        self.adata = adata
        self.filter_bank = filter_bank
        self.shape = shape
        self.cmap = cmap
        self.figsize = figsize
        self.img_alpha = img_alpha
        self.show_colorbar = show_colorbar

    # ------------------------------------------------------------------
    # Basic spatial scatter
    # ------------------------------------------------------------------

    def plot_spatial_scatter(
        self,
        gene_values: np.ndarray,
        ax: Optional[plt.Axes] = None,
        cmap: Optional[str | LinearSegmentedColormap] = None,
        figsize: Optional[tuple[int, int]] = None,
        size: Optional[float] = None,
        spines_color: str = B,
        title: str = "",
    ) -> plt.Axes:
        """
        Plots a spatial scatter of a gene (or any scalar per spot).

        Parameters
        ----------
        gene_values : np.ndarray
            One value per spot.
        ax : plt.Axes, optional
            Axis to draw on. If None, a new one is created.
        cmap : str or Colormap, optional
            Colormap to use. Defaults to the plotter's cmap.
        figsize : tuple, optional
            Figure size if a new figure is created.
        size : float, optional
            Spot size.
        spines_color : str
            Color of axis spines.
        title : str
            Plot title.

        Returns
        -------
        plt.Axes
            The axis with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)

        cmap = cmap or self.cmap

        self.adata.obs["__plot_values__"] = gene_values

        sq.pl.spatial_scatter(
            self.adata,
            shape=self.shape,
            color="__plot_values__",
            figsize=figsize or self.figsize,
            cmap=cmap,
            colorbar=self.show_colorbar,
            img_alpha=self.img_alpha,
            ax=ax,
            size=size,
        )

        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(title)

        for spine in ["bottom", "top", "right", "left"]:
            ax.spines[spine].set_color(spines_color)

        return ax

    # ------------------------------------------------------------------
    # MoG-transformed spatial map
    # ------------------------------------------------------------------

    def plot_mog_transformed_gene(
        self,
        gene_id: int,
        ax: Optional[plt.Axes] = None,
        cmap: LinearSegmentedColormap = BW_CMAP,
        figsize: Optional[tuple[int, int]] = None,
        mean_filter_it: int = SpatialFilterBank.DEFAULT_MEAN_FILTER_IT,
        exp_filter_it: int = SpatialFilterBank.DEFAULT_EXP_FILTER_IT,
    ) -> np.ndarray:
        """
        Applies the MoG transform to a gene and plots the binarized map.

        Parameters
        ----------
        gene_id : int
            Index of the gene in adata.var.
        ax : plt.Axes, optional
            Axis to draw on.
        cmap : Colormap
            Colormap for the binarized map.
        figsize : tuple, optional
            Figure size if a new figure is created.
        mean_filter_it : int
            Iterations for mean filter.
        exp_filter_it : int
            Iterations for expansion filter.

        Returns
        -------
        np.ndarray
            The transformed (binarized) gene values.
        """
        if self.filter_bank is None:
            raise RuntimeError("filter_bank is not set. Provide a SpatialFilterBank instance.")

        array_data = self.adata.X.toarray().T
        gene_data = array_data[gene_id].copy()

        transformed = self.filter_bank.transform_mog(
            gene_data,
            mean_filter_it=mean_filter_it,
            exp_filter_it=exp_filter_it,
        )

        self.plot_spatial_scatter(
            transformed,
            ax=ax,
            cmap=cmap,
            figsize=figsize or self.figsize,
            title=f"MoG-transformed gene {gene_id}",
        )

        return transformed

    # ------------------------------------------------------------------
    # Full diagnostic plot for a single gene
    # ------------------------------------------------------------------

    def full_gene_diagnostic_plot(
        self,
        gene_id: int,
        save: bool = False,
        path: Optional[str | Path] = None,
        mean_filter_it: int = SpatialFilterBank.DEFAULT_MEAN_FILTER_IT,
    ) -> None:
        """
        Recreates the professor's full diagnostic plot for a gene:
        - raw spatial map
        - filtered spatial map
        - histograms and CDFs
        - MoG-transformed spatial map

        Parameters
        ----------
        gene_id : int
            Index of the gene in adata.var.
        save : bool
            Whether to save the figure.
        path : str or Path, optional
            Path to save the figure if save=True.
        mean_filter_it : int
            Iterations for mean filter.
        """
        if self.filter_bank is None:
            raise RuntimeError("filter_bank is not set. Provide a SpatialFilterBank instance.")

        array_data = self.adata.X.toarray().T
        gene_data = array_data[gene_id].copy()
        gene_name = self.adata.var.iloc[[gene_id]].index[0]

        title = f"id: {gene_id}, name: {gene_name}"

        fig = plt.figure(figsize=(25, 10), facecolor=B, dpi=100)
        gs = fig.add_gridspec(2, 5)

        ax11 = fig.add_subplot(gs[0, 0])
        ax12 = fig.add_subplot(gs[0, 1])
        ax13 = fig.add_subplot(gs[0, 2])

        ax21 = fig.add_subplot(gs[1, 0])
        ax22 = fig.add_subplot(gs[1, 1])
        ax23 = fig.add_subplot(gs[1, 2])

        aximg = fig.add_subplot(gs[:, 3:])

        fig.suptitle(title, fontsize=20)

        # --- ax12: raw spatial scatter ---
        self.plot_spatial_scatter(gene_data, ax=ax12, title="Raw spatial")

        # --- ax22: mean-filtered spatial scatter ---
        filtered_gene_data = self.filter_bank.mean_filter(gene_data.copy(), mean_filter_it)
        self.plot_spatial_scatter(filtered_gene_data, ax=ax22, title="Mean-filtered")

        # --- ax11: distribution of non-zero values ---
        near_zero_idx = gene_data < 0.4
        near_zero_perc = near_zero_idx.astype(int).sum() / len(gene_data)
        gene_data_not_zero = gene_data[~near_zero_idx]

        ax11.hist(gene_data_not_zero, bins="auto", density=True)
        ax11.set_title(f"near 0 perc.: {near_zero_perc * 100:.2f}%")
        ax11.spines["top"].set_color(B)
        ax11.spines["right"].set_color(B)

        # --- ax21: ordered CDF of non-zero values ---
        ax21.scatter(
            range(len(gene_data_not_zero)),
            np.sort(gene_data_not_zero),
            s=10,
            alpha=0.05,
        )
        ax21.set_ylim([0, int(gene_data.max()) + 1])
        ax21.spines["top"].set_color(B)
        ax21.spines["right"].set_color(B)
        ax21.set_title("Ordered CDF (non-zero values)")

        # --- ax13: histogram + GMM fit of filtered data ---
        filtered_gene_data = self.filter_bank.mean_filter(gene_data.copy(), mean_filter_it)
        ax13.hist(filtered_gene_data, bins="auto", density=True)
        ax13.spines["top"].set_color(B)
        ax13.spines["right"].set_color(B)
        ax13.set_title("Filtered distribution + GMM fit")

        gm = GaussianMixture(n_components=2, random_state=0).fit(filtered_gene_data.reshape(-1, 1))
        m1, m2 = gm.means_.flatten()
        w1, w2 = gm.weights_
        s1, s2 = np.sqrt(gm.covariances_).flatten()

        # Ensure m1 < m2
        if m1 > m2:
            m1, m2 = m2, m1
            s1, s2 = s2, s1
            w1, w2 = w2, w1

        minx = filtered_gene_data.min()
        maxx = filtered_gene_data.max()
        x = np.linspace(minx - 0.5, maxx + 0.5, 200)
        g1y = norm(m1, s1).pdf(x) * w1
        g2y = norm(m2, s2).pdf(x) * w2

        ax13.axvline(m1 + 2.5 * s1, color="r", alpha=0.5)
        ax13.axvline(m1 - 2.5 * s1, color="r", alpha=0.5)
        ax13.plot(x, g1y, "r", alpha=0.5)
        ax13.plot(x, g2y, "y", alpha=0.5)

        # --- ax23: ordered CDF of filtered data ---
        ax23.scatter(
            range(len(filtered_gene_data)),
            np.sort(filtered_gene_data),
            s=10,
            alpha=0.05,
        )
        ax23.set_ylim([0, int(filtered_gene_data.max()) + 1])
        ax23.spines["top"].set_color(B)
        ax23.spines["right"].set_color(B)
        ax23.set_title("Ordered CDF (filtered values)")

        # --- aximg: MoG-transformed spatial map ---
        self.plot_mog_transformed_gene(
            gene_id,
            ax=aximg,
            cmap=BW_CMAP,
            figsize=(10, 10),
        )
        aximg.set_title("MoG-transformed spatial map")

        if save and path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, format="png", bbox_inches="tight")
        plt.show()
        plt.close(fig)
