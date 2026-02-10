from __future__ import annotations
import scanpy as sc
import numpy as np
from pathlib import Path
import platform
from typing import Optional, List

from src.preprocessing.spatial_filters import SpatialFilterBank

class SpatialDataset:
    """
    Wrapper around a 10x Genomics Visium dataset.
    Handles loading, preprocessing, and basic gene-level utilities.
    """

    def __init__(self, dataset_path: str | Path):
        """
        Parameters
        ----------
        dataset_path : str or Path
            Path to the folder containing the Visium dataset.
            Example: "data/DLPFC-151673"
        """
        self.dataset_path = Path(dataset_path)
        self.adata = None
        self.filter_bank: Optional[SpatialFilterBank] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, load_images: bool = True) -> None:
        """
        Loads the Visium dataset using Scanpy and initializes spatial filters.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.dataset_path}")

        self.adata = sc.read_visium(
            str(self.dataset_path),
            count_file="filtered_feature_bc_matrix.h5",
            load_images=load_images
        )

        self.adata.var_names_make_unique()

        # Initialize spatial filters (neighborhood graph)
        self.filter_bank = SpatialFilterBank(self.adata)

    # ------------------------------------------------------------------
    # Gene utilities
    # ------------------------------------------------------------------

    def get_gene_expression(self, gene_id: int) -> np.ndarray:
        """
        Returns the expression vector of a gene across all spots.

        Parameters
        ----------
        gene_id : int
            Index of the gene in adata.var

        Returns
        -------
        np.ndarray
            Expression values for each spot.
        """
        if self.adata is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        return self.adata.X.toarray().T[gene_id]

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    def select_top_spatially_variable_genes(
        self,
        n_top: int,
        min_gene_expression: int = 300,
        n_top_genes: int = 3000
    ) -> np.ndarray:
        """
        Selects the top spatially variable genes using Scanpy HVG ranking
        with additional filtering based on total expression.

        Parameters
        ----------
        n_top : int
            Number of genes to return.
        min_gene_expression : int
            Minimum total expression required to keep a gene.
        n_top_genes : int
            Number of genes considered by Scanpy HVG ranking.

        Returns
        -------
        np.ndarray
            Array of gene indices.
        """
        if self.adata is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        data = self.adata

        # Total expression per gene
        tot_gene_expression = data.X.toarray().sum(axis=0)

        valid_mask = np.isfinite(tot_gene_expression) & (tot_gene_expression > 0)
        if not np.any(valid_mask): 
            raise ValueError("No valid genes found (all zero or non-finite).") 
        
        data_hvg = data[:, valid_mask].copy() 

        # -------------------------------
        # OS-dependent HVG selection
        # -------------------------------
        system = platform.system()

        if system == "Windows":
            hvg_flavor = "seurat"
            sc.pp.normalize_total(data_hvg, target_sum=1e4)
            sc.pp.log1p(data_hvg)
        else:
            hvg_flavor = "seurat_v3"

        print(f"[HVG] Using flavor='{hvg_flavor}' on {system}")

        sc.pp.highly_variable_genes(
            data_hvg,
            flavor=hvg_flavor,
            n_top_genes=n_top_genes
        )

        if "highly_variable_rank" not in data_hvg.var.columns:
            data_hvg.var["highly_variable_rank"] = (
                data_hvg.var["dispersions_norm"]
                .rank(ascending=False, method="min") - 1
            )

        # HVG ranking on the filtered subset
        var_rank_sub = data_hvg.var["highly_variable_rank"].to_numpy()
        var_rank_sub = np.nan_to_num(var_rank_sub, nan=np.nanmax(var_rank_sub) + 1)
        max_rank_sub = np.nanmax(var_rank_sub)

        # Lift ranks back to full gene space
        var_rank_full = np.full(data.n_vars, max_rank_sub, dtype=float)
        idx_full = np.where(valid_mask)[0]
        var_rank_full[idx_full] = var_rank_sub

        # Penalize low-expression genes in the full space
        low_expr_idx = np.where(tot_gene_expression < min_gene_expression)[0]
        var_rank_full[low_expr_idx] = max_rank_sub

        # Select top genes
        smallest = np.argsort(var_rank_full)[:n_top]
        smallest = smallest[(var_rank_full[smallest] != max_rank_sub)]

        return smallest
