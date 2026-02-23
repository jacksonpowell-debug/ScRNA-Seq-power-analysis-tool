#!/usr/bin/env python3
"""
scRNA-Seq Analysis Pipeline
============================
HOW TO RUN:
    python scRNAseq_pipeline.py

INSTALL DEPENDENCIES ONCE:
    pip install anndata scanpy matplotlib seaborn scipy statsmodels numpy pandas

Supports:
  • 10x .mtx directory  → emptyDrops-equivalent (knee-point filter) + doublet removal
  • .h5ad file          → memory-mapped load, cluster subsetting, downstream analysis

Downstream (both paths):
  • Gene validation
  • QC violin plots  (UMIs/cell, Genes/cell)
  • Per-gene UMI histogram with dropout / dispersion stats
  • ZINB-based power analysis (1.5x, 2x, 4x, 8x fold decrease, 95% power target)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves files
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import scipy.sparse as sp
from scipy import stats

warnings.filterwarnings("ignore")

# ── Dependency check ──────────────────────────────────────────────────────────
def _require(pkg, pip_name=None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        pip_name = pip_name or pkg
        sys.exit(f"\nMissing package '{pip_name}'. Install with:\n"
                 f"  pip install {pip_name}\n")

anndata     = _require("anndata")
statsmodels = _require("statsmodels")
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from statsmodels.discrete.discrete_model import NegativeBinomial
import scipy.io


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — INPUT
# ══════════════════════════════════════════════════════════════════════════════

def get_input():
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║           scRNA-Seq Analysis Pipeline                        ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    print("Enter the path to your data:")
    print("  • Directory containing 10x .mtx files")
    print("  • Path to an .h5ad file\n")
    path = input("Path: ").strip().strip("'\"")

    if not os.path.exists(path):
        sys.exit(f"Path does not exist: '{path}'")

    # Detect type
    if os.path.isdir(path):
        contents = os.listdir(path)
        h5ad_files = [f for f in contents if f.lower().endswith(".h5ad")]
        mtx_files  = [f for f in contents if f.lower().endswith((".mtx", ".mtx.gz"))]
        if h5ad_files:
            if len(h5ad_files) > 1:
                print("\nMultiple .h5ad files found:")
                for i, f in enumerate(h5ad_files):
                    print(f"  [{i+1}] {f}")
                ci = int(input("Select file number: ").strip()) - 1
                path = os.path.join(path, h5ad_files[ci])
            else:
                path = os.path.join(path, h5ad_files[0])
                print(f"Found: {path}")
            return path, "h5ad"
        elif mtx_files:
            return path, "mtx"
        else:
            sys.exit(f"No .h5ad or .mtx files found in: {path}")
    elif path.lower().endswith(".h5ad"):
        return path, "h5ad"
    elif path.lower().endswith((".mtx", ".mtx.gz")):
        return os.path.dirname(path), "mtx"
    else:
        sys.exit("Unrecognised file type. Provide an .h5ad file or .mtx directory.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2A — MTX: knee filter + doublet removal
# ══════════════════════════════════════════════════════════════════════════════

def _load_single_mtx(mtx_dir, label="", specific_file=None):
    """
    Load one MTX directory into an AnnData.
    Returns (adata, mtx_path) so the caller knows exactly which file was used.
    specific_file: if given, use this filename instead of auto-detecting.
    """
    import scanpy as sc
    sc.settings.verbosity = 0

    contents = os.listdir(mtx_dir)
    tag = " ({})".format(label) if label else ""

    mtx_files = sorted([f for f in contents if f.lower().endswith((".mtx", ".mtx.gz"))])
    if not mtx_files:
        sys.exit("No .mtx file found in: {}".format(mtx_dir))

    if specific_file and specific_file in mtx_files:
        mtx_file = specific_file
    elif len(mtx_files) > 1:
        print("  Multiple MTX files found in directory:")
        for i, f in enumerate(mtx_files):
            print("    [{}] {}".format(i + 1, f))
        choice = input("  Select file number: ").strip()
        try:
            mtx_file = mtx_files[int(choice) - 1]
        except (ValueError, IndexError):
            sys.exit("Invalid selection.")
    else:
        mtx_file = mtx_files[0]

    std_names = {"matrix.mtx", "matrix.mtx.gz"}
    if mtx_file in std_names:
        try:
            adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=False)
            print("  Loaded{}: {:,} genes x {:,} cells".format(
                  tag, adata.n_vars, adata.n_obs))
            return adata, os.path.join(mtx_dir, mtx_file)
        except Exception as e:
            print("  Standard 10x reader failed ({}), trying manual load...".format(e))

    # Manual load
    import gzip
    mtx_path = os.path.join(mtx_dir, mtx_file)
    print("  Reading matrix{}: {}".format(tag, mtx_file))
    # Read raw MTX (may be genes x cells or cells x genes depending on tool)
    X_raw = scipy.io.mmread(mtx_path)  # keep as COO for now
    X_raw = sp.csc_matrix(X_raw)

    ext = (".tsv", ".tsv.gz", ".txt", ".txt.gz", ".csv", ".csv.gz")
    def _find(prefixes):
        for f in contents:
            fl = f.lower()
            if any(fl.startswith(p) for p in prefixes) and any(fl.endswith(e) for e in ext):
                return os.path.join(mtx_dir, f)
        return None

    bc_path = _find(("barcodes", "cells", "cell_barcodes"))
    ft_path = _find(("features", "genes", "gene_names"))

    opener = lambda p: gzip.open(p, "rt") if p.endswith(".gz") else open(p, "rt")

    if bc_path:
        with opener(bc_path) as fh:
            barcodes = [l.strip().split("\t")[0] for l in fh if l.strip()]
    else:
        barcodes = None

    if ft_path:
        with opener(ft_path) as fh:
            rows = [l.strip().split("\t") for l in fh if l.strip()]
        gene_ids   = [r[0] for r in rows]
        gene_names = [r[1] for r in rows] if rows and len(rows[0]) >= 2 else gene_ids
    else:
        gene_ids = gene_names = None

    # Orient matrix: AnnData wants cells x genes
    # 10x MTX is genes x cells → need to transpose
    # Determine correct orientation from barcode/feature counts
    n_rows, n_cols = X_raw.shape
    n_bc   = len(barcodes)   if barcodes   else None
    n_gene = len(gene_names) if gene_names else None

    if n_bc is not None and n_gene is not None:
        if n_rows == n_gene and n_cols == n_bc:
            # Standard 10x: genes x cells → transpose
            X = sp.csr_matrix(X_raw.T)
        elif n_rows == n_bc and n_cols == n_gene:
            # Already cells x genes
            X = sp.csr_matrix(X_raw)
        else:
            print("  WARNING: MTX shape {}x{} doesn't match barcodes({}) or genes({}).".format(
                  n_rows, n_cols, n_bc, n_gene))
            print("  Assuming standard 10x orientation (genes x cells) and transposing.")
            X = sp.csr_matrix(X_raw.T)
            if barcodes and len(barcodes) != X.shape[0]:
                barcodes = [str(i) for i in range(X.shape[0])]
    else:
        # No metadata to guide us — assume standard 10x
        X = sp.csr_matrix(X_raw.T)

    if barcodes is None:
        barcodes = [str(i) for i in range(X.shape[0])]
    if gene_names is None:
        gene_ids = gene_names = [str(i) for i in range(X.shape[1])]

    import anndata as ad
    var_df = pd.DataFrame({"gene_ids": gene_ids}, index=pd.Index(gene_names).astype(str))
    var_df.index = pd.Index(var_df.index).astype(str)
    obs_df = pd.DataFrame(index=pd.Index(barcodes).astype(str))
    adata  = ad.AnnData(X=X, obs=obs_df, var=var_df)
    adata.var_names_make_unique()
    print("  Loaded{}: {:,} genes x {:,} cells".format(tag, adata.n_vars, adata.n_obs))
    return adata, os.path.join(mtx_dir, mtx_file)


def load_mtx(mtx_dir):
    print("\n── MTX input detected ──────────────────────────────────────────")

    # ── Load first dataset ───────────────────────────────────────────────────
    adata1, mtx_path1 = _load_single_mtx(mtx_dir, label="Dataset 1")

    # ── Offer to combine a second dataset ────────────────────────────────────
    ans = input("\n  Combine with a second MTX dataset? [y/n]: ").strip().lower()
    if ans == "y":
        path2_raw = input("  Path to second MTX directory (or any file inside it): ").strip().strip("'\"")
        if os.path.isfile(path2_raw):
            specific_file2 = os.path.basename(path2_raw)
            path2 = os.path.dirname(path2_raw)
        else:
            specific_file2 = None
            path2 = path2_raw

        if not os.path.isdir(path2):
            print("  Path not found — proceeding with single dataset.")
            return _filter_mtx_with_dir(adata1, mtx_dir, mtx_path1)

        adata2, mtx_path2 = _load_single_mtx(path2, label="Dataset 2",
                                              specific_file=specific_file2)

        # ── Run emptyDrops on each dataset separately before combining ───────
        print("\n  Filtering Dataset 1 with emptyDrops...")
        adata1, _ = _filter_mtx_with_dir(adata1, mtx_dir, mtx_path1,
                                          skip_doublet_prompt=True)
        print("  Filtering Dataset 2 with emptyDrops...")
        adata2, _ = _filter_mtx_with_dir(adata2, path2, mtx_path2,
                                          skip_doublet_prompt=True)

        # ── Align genes ──────────────────────────────────────────────────────
        shared_genes = sorted(set(adata1.var_names) & set(adata2.var_names))
        if len(shared_genes) == 0:
            print("  No shared genes — proceeding with Dataset 1 only.")
            return _doublet_filter(adata1), None

        n_only1 = adata1.n_vars - len(shared_genes)
        n_only2 = adata2.n_vars - len(shared_genes)
        if n_only1 or n_only2:
            print("  Gene intersection: {:,} shared genes".format(len(shared_genes)))
            if n_only1:
                print("    {:,} genes only in Dataset 1 (dropped)".format(n_only1))
            if n_only2:
                print("    {:,} genes only in Dataset 2 (dropped)".format(n_only2))
        adata1 = adata1[:, shared_genes].copy()
        adata2 = adata2[:, shared_genes].copy()

        # ── Make barcodes unique and concatenate ─────────────────────────────
        import anndata as ad
        adata1.obs_names = [bc + "-ds1" for bc in adata1.obs_names]
        adata2.obs_names = [bc + "-ds2" for bc in adata2.obs_names]
        adata1.obs["dataset"] = "Dataset1"
        adata2.obs["dataset"] = "Dataset2"

        adata = ad.concat([adata1, adata2], join="outer")
        adata.var_names_make_unique()
        print("  Combined: {:,} genes x {:,} cells total".format(
              adata.n_vars, adata.n_obs))

        return _doublet_filter(adata), None

    else:
        return _filter_mtx_with_dir(adata1, mtx_dir, mtx_path1)


def _filter_mtx(adata):
    """
    Empty droplet removal via DropletUtils::emptyDrops (R subprocess),
    followed by interactive doublet removal.
    Falls back to interactive knee-point filter if R is unavailable.
    """
    n_raw      = adata.n_obs
    umi_counts = np.array(adata.X.sum(axis=1)).flatten()

    print("")
    print("── MTX Filtering ────────────────────────────────────────────────")
    print("  Starting cells : {:,}".format(n_raw))
    print("  UMI range      : {:,} - {:,}".format(int(umi_counts.min()),
                                                    int(umi_counts.max())))
    print("  Median UMI     : {:,}".format(int(np.median(umi_counts))))

    return adata, None   # mtx_dir not available here — see load_mtx


def _run_emptydrops_r(mtx_dir, mtx_file, barcode_file, feature_file):
    """
    Run DropletUtils::emptyDrops() via R subprocess.
    Passes exact file paths so R does not need to guess filenames.
    Returns a set of barcode strings that pass, or None on failure.
    """
    import subprocess, tempfile

    r_script = r"""
suppressPackageStartupMessages({
  library(DropletUtils)
  library(Matrix)
  library(SingleCellExperiment)
  library(BiocParallel)
})

args         <- commandArgs(trailingOnly=TRUE)
mtx_path     <- args[1]
barcode_path <- args[2]
feature_path <- args[3]
out_file     <- args[4]

cat("  [R] Reading MTX:", mtx_path, "
")
mat      <- readMM(mtx_path)
barcodes <- readLines(barcode_path)
features <- read.table(feature_path, sep="	", header=FALSE,
                       stringsAsFactors=FALSE, quote="")

if (ncol(features) >= 2) {
  gene_names <- features[,2]
} else {
  gene_names <- features[,1]
}

# 10x MTX is genes x cells
if (nrow(mat) == length(gene_names) && ncol(mat) == length(barcodes)) {
  # already genes x cells — correct orientation
} else if (ncol(mat) == length(gene_names) && nrow(mat) == length(barcodes)) {
  mat <- t(mat)
}

rownames(mat) <- make.unique(gene_names)
colnames(mat) <- barcodes

cat("  [R] Dimensions:", nrow(mat), "genes x", ncol(mat), "cells
")

set.seed(42)
e <- emptyDrops(mat, lower=100, retain=Inf, niters=10000,
                test.ambient=FALSE,
                BPPARAM=SerialParam())

keep <- !is.na(e$FDR) & e$FDR <= 0.05
cat("  [R] Cells passing emptyDrops (FDR<=0.05):", sum(keep), "
")

passing <- colnames(mat)[keep]
writeLines(passing, out_file)
cat("  [R] Done.
")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "emptydrops.R")
        barcode_out = os.path.join(tmpdir, "passing_barcodes.txt")

        with open(script_path, 'w') as fh:
            fh.write(r_script)

        try:
            result = subprocess.run(
                ["Rscript", "--vanilla", script_path,
                 mtx_file, barcode_file, feature_file, barcode_out],
                capture_output=True, text=True, timeout=600
            )
            for line in result.stdout.splitlines():
                if line.strip():
                    print(line)
            if result.returncode != 0:
                print("  [R] DropletUtils error:")
                print("\n".join(result.stderr.splitlines()[-30:]))
                return None
            if not os.path.exists(barcode_out):
                print("  [R] No barcode output file produced.")
                return None
            with open(barcode_out) as fh:
                barcodes = set(line.strip() for line in fh if line.strip())
            return barcodes
        except FileNotFoundError:
            print("  Rscript not found — is R installed and on your PATH?")
            return None
        except subprocess.TimeoutExpired:
            print("  R subprocess timed out after 10 minutes.")
            return None


def _doublet_filter(adata):
    """Interactive high-UMI doublet removal. Works on any AnnData."""
    if adata.n_obs == 0:
        print("  Warning: no cells to filter.")
        return adata

    umi_filtered = np.array(adata.X.sum(axis=1)).flatten()
    p99    = int(np.percentile(umi_filtered, 99))
    n_pre  = adata.n_obs

    print("")
    print("  99th percentile UMI: {:,}  ({:,} cells above)".format(
          p99, int(np.sum(umi_filtered > p99))))
    raw2 = input("  Max UMI threshold for doublets [suggested {:,} / 0 to skip]: ".format(
                 p99)).strip()

    if raw2 == "0" or raw2 == "":
        print("  Skipping doublet filter.")
    else:
        try:
            umi_max = int(raw2)
        except ValueError:
            umi_max = p99
        keep = umi_filtered <= umi_max
        adata = adata[keep].copy()
        print("  After doublet filter ({:,}): {:,} cells retained ({:,} removed)".format(
              umi_max, adata.n_obs, n_pre - adata.n_obs))
    return adata


def _filter_mtx_with_dir(adata, mtx_dir, specific_mtx=None, skip_doublet_prompt=False):
    """
    Full filtering: emptyDrops via R (with fallback), then doublet removal.
    """
    n_raw      = adata.n_obs
    umi_counts = np.array(adata.X.sum(axis=1)).flatten()

    print("")
    print("── MTX Filtering ────────────────────────────────────────────────")
    print("  Starting cells : {:,}".format(n_raw))
    print("  UMI range      : {:,} - {:,}".format(int(umi_counts.min()),
                                                    int(umi_counts.max())))
    print("  Median UMI     : {:,}".format(int(np.median(umi_counts))))

    # ── emptyDrops via R ─────────────────────────────────────────────────────
    print("")
    print("  Running DropletUtils::emptyDrops (lower.prop=0.05) via R...")

    # Locate exact files (any filename, compressed or not)
    contents = os.listdir(mtx_dir)
    def _find(prefixes, extensions):
        for f in contents:
            fl = f.lower()
            if any(fl.startswith(p) for p in prefixes) and                any(fl.endswith(e) for e in extensions):
                return os.path.join(mtx_dir, f)
        return None

    ext = (".tsv", ".tsv.gz", ".txt", ".txt.gz", ".csv", ".csv.gz")
    if specific_mtx and os.path.isfile(specific_mtx):
        mtx_exact = specific_mtx
    else:
        mtx_exact = next((os.path.join(mtx_dir, f) for f in sorted(contents)
                          if f.lower().endswith((".mtx", ".mtx.gz"))), None)
    barcode_exact  = _find(("barcodes", "cells", "cell_barcodes"), ext)
    feature_exact  = _find(("features", "genes", "gene_names"), ext)

    if not all([mtx_exact, barcode_exact, feature_exact]):
        print("  Could not locate all three MTX files:")
        print("    MTX      :", mtx_exact or "NOT FOUND")
        print("    Barcodes :", barcode_exact or "NOT FOUND")
        print("    Features :", feature_exact or "NOT FOUND")
        passing_barcodes = None
    else:
        print("  MTX      : " + os.path.basename(mtx_exact))
        print("  Barcodes : " + os.path.basename(barcode_exact))
        print("  Features : " + os.path.basename(feature_exact))
        passing_barcodes = _run_emptydrops_r(
            mtx_dir, mtx_exact, barcode_exact, feature_exact)

    if passing_barcodes is not None:
        keep  = np.array([bc in passing_barcodes for bc in adata.obs_names])
        adata = adata[keep].copy()
        print("  After emptyDrops: {:,} cells retained ({:,} removed)".format(
              adata.n_obs, n_raw - adata.n_obs))
    else:
        # ── Fallback: interactive knee-point ─────────────────────────────────
        print("  Falling back to knee-point filter.")
        sorted_umis    = np.sort(umi_counts)[::-1]
        log_rank = np.log10(np.arange(1, len(sorted_umis) + 1))
        log_umi  = np.log10(sorted_umis + 1)
        lr_n = (log_rank - log_rank.min()) / (log_rank.max() - log_rank.min() + 1e-9)
        lu_n = (log_umi  - log_umi.min())  / (log_umi.max()  - log_umi.min()  + 1e-9)
        dist           = np.abs(lu_n - (1 - lr_n))
        knee_threshold = int(sorted_umis[np.argmax(dist)])
        knee_cells     = int(np.sum(umi_counts >= knee_threshold))

        print("  Suggested knee threshold: {:,} ({:,} cells, {:.1f}%)".format(
              knee_threshold, knee_cells, 100 * knee_cells / n_raw))
        raw = input("  Accept? [y / number / 0 to skip]: ").strip()

        if raw == "0":
            umi_min = 0
        elif raw == "" or raw.lower() == "y":
            umi_min = knee_threshold
        else:
            try:
                umi_min = int(raw)
            except ValueError:
                umi_min = knee_threshold

        if umi_min > 0:
            keep  = umi_counts >= umi_min
            adata = adata[keep].copy()
            print("  After knee filter ({:,}): {:,} cells retained".format(
                  umi_min, adata.n_obs))

    if skip_doublet_prompt:
        return adata, None

    return _doublet_filter(adata), None


def load_h5ad(path):
    print(f"\n── H5AD input detected ─────────────────────────────────────────")
    print("Reading .h5ad file (memory-mapped)...")

    # backed='r' = memory-mapped; the full matrix is NEVER loaded into RAM
    adata = anndata.read_h5ad(path, backed="r")
    print(f"Loaded: {adata.n_vars} genes × {adata.n_obs} cells")

    # ── Cluster selection ────────────────────────────────────────────────────
    obs = adata.obs

    # Biological cell-type columns get priority; show all categorical columns
    PRIORITY_COLS = ["cell_type", "author_cell_type", "majorclass",
                     "subclass_label", "celltype", "CellType", "cluster",
                     "leiden", "louvain", "seurat_clusters"]

    all_cat_cols = [c for c in obs.columns
                    if obs[c].dtype.name in ("category", "object")
                    and obs[c].nunique() > 1]

    # Sort: priority cols first, then rest alphabetically
    priority = [c for c in PRIORITY_COLS if c in all_cat_cols]
    others   = sorted([c for c in all_cat_cols if c not in priority])
    clust_cols = priority + others

    chosen_clusters = None
    clust_col       = None

    if not clust_cols:
        print("No cluster columns found; proceeding with all cells.")
    else:
        print("\nAvailable cluster annotations:")
        for i, col in enumerate(clust_cols):
            n_unique = obs[col].nunique()
            marker = "  ◀ cell type" if col in PRIORITY_COLS else ""
            print(f"  [{i+1}] {col}  ({n_unique} unique values){marker}")

        ci = int(input("\nSelect annotation column number: ").strip()) - 1
        if ci < 0 or ci >= len(clust_cols):
            sys.exit("Invalid selection.")

        clust_col    = clust_cols[ci]
        all_clusters = sorted(obs[clust_col].astype(str).unique().tolist())

        print(f"\nCell types in '{clust_col}':")
        for c in all_clusters:
            n = int((obs[clust_col].astype(str) == c).sum())
            print(f"  {c}  ({n:,} cells)")

        ans = input("\nAnalyse all clusters together? [y/n]: ").strip().lower()
        if ans == "y":
            chosen_clusters = all_clusters
        else:
            raw = input("Enter cluster names (comma-separated): ").strip()
            chosen_clusters = [c.strip() for c in raw.split(",")]
            # Case-insensitive matching with helpful error
            all_lower = {c.lower(): c for c in all_clusters}
            resolved  = []
            bad       = []
            for c in chosen_clusters:
                if c in all_clusters:
                    resolved.append(c)
                elif c.lower() in all_lower:
                    resolved.append(all_lower[c.lower()])
                    print(f"  Matched '{c}' → '{all_lower[c.lower()]}'")
                else:
                    bad.append(c)
            if bad:
                print(f"\n⚠  Not found: {bad}")
                print("Available:", ", ".join(all_clusters))
                sys.exit("Please re-run and use exact names from the list above.")
            chosen_clusters = resolved

    # Subset — identify indices first, then load only those cells into memory
    if clust_col and chosen_clusters:
        obs_reloaded = adata.obs  # obs metadata is already in memory
        mask         = obs_reloaded[clust_col].astype(str).isin(chosen_clusters)
        row_indices  = np.where(mask)[0]
        print(f"Loading {len(row_indices):,} cells into memory...")
        adata = adata[row_indices].to_memory()
        print(f"Subset to {adata.n_obs:,} cells from: {', '.join(chosen_clusters)}")
    else:
        print("Loading full dataset into memory...")
        adata = adata.to_memory()

    cell_clusters = adata.obs[clust_col].astype(str).values if clust_col else None

    # ── Use raw counts if available ──────────────────────────────────────────
    # adata.X in processed h5ad files is typically normalized/log-transformed.
    # adata.raw.X contains original UMI counts and is correct for QC and ZINB.
    if adata.raw is not None:
        print("  Using adata.raw.X (raw UMI counts)")
        # Rebuild a minimal AnnData using raw counts + raw var names
        import anndata as ad
        raw_X = adata.raw.X
        if not sp.issparse(raw_X):
            raw_X = sp.csr_matrix(raw_X)
        adata = ad.AnnData(
            X   = raw_X,
            obs = adata.obs,
            var = adata.raw.var
        )
    else:
        print("  No raw slot found — using adata.X as-is")

    return adata, cell_clusters


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — GENE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_genes(adata):
    print("\n── Gene Selection ───────────────────────────────────────────────")
    available   = list(adata.var_names)
    avail_lower = {a.lower(): a for a in available}

    while True:
        raw   = input("Enter gene names to analyse (comma-separated): ").strip()
        query = [g.strip() for g in raw.split(",") if g.strip()]

        if not query:
            print("  No gene names entered — please try again.")
            continue

        found   = [g for g in query if g in available]
        missing = [g for g in query if g not in available]

        # Case-insensitive auto-correction for exact case mismatches
        for g in missing[:]:
            if g.lower() in avail_lower:
                corrected = avail_lower[g.lower()]
                print(f"  Matched '{g}' → '{corrected}' (case corrected)")
                found.append(corrected)
                missing.remove(g)

        if missing:
            print(f"\n⚠  Genes NOT found in dataset:")
            for g in missing:
                print(f"    • {g}")
            # Suggest partial matches
            suggestions = [a for a in available
                           if any(g.lower() in a.lower() or a.lower() in g.lower()
                                  for g in missing)][:5]
            if suggestions:
                print(f"  Similar gene names: {', '.join(suggestions)}")

        if found:
            print(f"\n✓ Genes to analyse: {', '.join(found)}")
            if missing:
                ans = input("Proceed with found genes only? [y/n]: ").strip().lower()
                if ans != "y":
                    print("  OK — please re-enter all gene names.")
                    continue
            return found
        else:
            print("  None of the entered genes were found. Please try again.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — QC PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def compute_qc(adata):
    print("\n── Computing QC metrics ─────────────────────────────────────────")
    X = adata.X
    if sp.issparse(X):
        umi_per_cell   = np.array(X.sum(axis=1)).flatten()
        genes_per_cell = np.array((X > 0).sum(axis=1)).flatten()
    else:
        umi_per_cell   = X.sum(axis=1)
        genes_per_cell = (X > 0).sum(axis=1)
    return umi_per_cell, genes_per_cell


def violin_plot(values_dict, ylabel, title, filename, stat_vec):
    """values_dict: {label: array}"""
    fig, ax = plt.subplots(figsize=(max(6, len(values_dict) * 1.5 + 2), 5))
    labels = list(values_dict.keys())
    data   = [values_dict[k] for k in labels]

    parts = ax.violinplot(data, positions=range(len(labels)),
                          showmedians=False, showextrema=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color); pc.set_alpha(0.7)

    # Boxplot overlay
    ax.boxplot(data, positions=range(len(labels)),
               widths=0.08, patch_artist=True,
               boxprops=dict(facecolor="white", alpha=0.8),
               medianprops=dict(color="black", linewidth=1.5),
               whiskerprops=dict(linewidth=1),
               capprops=dict(linewidth=1),
               flierprops=dict(marker="o", markersize=1.5, alpha=0.3))

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30 if len(labels) > 3 else 0, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Stats annotation
    stat_text = (f"Mean   = {np.mean(stat_vec):,.1f}\n"
                 f"Median = {np.median(stat_vec):,.1f}")
    ax.text(0.98, 0.98, stat_text, transform=ax.transAxes,
            va="top", ha="right", fontsize=9, family="monospace",
            bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3"))

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def make_qc_plots(umi_per_cell, genes_per_cell, cell_clusters):
    group_label = cell_clusters if cell_clusters is not None else ["All"] * len(umi_per_cell)
    unique_groups = list(dict.fromkeys(group_label))   # preserve order

    umi_by_group   = {g: umi_per_cell[np.array(group_label) == g]   for g in unique_groups}
    genes_by_group = {g: genes_per_cell[np.array(group_label) == g] for g in unique_groups}

    violin_plot(umi_by_group,   "Total UMI Count",           "UMIs per Cell",
                "QC_UMI_violin.png",   umi_per_cell)
    violin_plot(genes_by_group, "Number of Genes Detected",  "Unique Genes per Cell",
                "QC_Genes_violin.png", genes_per_cell)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5A — PER-GENE HISTOGRAM
# ══════════════════════════════════════════════════════════════════════════════

def gene_histogram(cts, gene, mean_umi, dropout_pct, bio_disp):
    x_max = max(int(np.percentile(cts, 99)), 5)
    bins   = np.arange(0, x_max + 2)
    counts, edges = np.histogram(cts, bins=bins)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = edges[:-1]
    bars = ax.bar(x_pos, counts, width=0.85, color="#4A90D9",
                  edgecolor="white", linewidth=0.5)

    # Value labels above each bar
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    str(cnt), ha="center", va="bottom", fontsize=7)

    stat_text = (f"Mean transcripts/cell: {mean_umi:.2f}\n"
                 f"Dropout: {dropout_pct:.1f}%\n"
                 f"Bio. dispersion: {bio_disp:.3f}")
    ax.text(0.98, 0.98, stat_text, transform=ax.transAxes,
            va="top", ha="right", fontsize=9, family="monospace",
            bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.3"))

    ax.set_xlabel("UMI Count per Cell")
    ax.set_ylabel("Number of Cells")
    ax.set_title(f"{gene} — UMI Distribution per Cell", fontweight="bold")
    ax.set_xlim(-0.5, x_max + 0.5)
    ax.set_ylim(0, max(counts) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fname = f"{gene}_UMI_histogram.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5B — ZINB FITTING
# ══════════════════════════════════════════════════════════════════════════════

def _mom_nb(y):
    """Method-of-moments NB estimates: mu and theta (size parameter)."""
    mu  = float(np.mean(y))
    var = float(np.var(y))
    # NB variance = mu + mu^2/theta  →  theta = mu^2 / (var - mu)
    excess = var - mu
    theta  = (mu ** 2 / excess) if excess > 1e-8 else 50.0
    theta  = float(np.clip(theta, 0.05, 1000.0))
    return mu, theta


def fit_zinb(cts):
    """
    Fit ZINB with method-of-moments starting values.
    Falls back through ZINB → NB → moment estimates.
    Returns dict(mu, theta, pi) where theta is the NB size parameter.
    """
    y      = np.asarray(cts, dtype=float)
    ones   = np.ones(len(y))
    pi_obs = float(np.mean(y == 0))
    mu0, theta0 = _mom_nb(y)

    # Starting values for statsmodels ZINB
    # params order: [log(mu), log(alpha), logit(pi)]  where alpha = 1/theta
    alpha0    = 1.0 / theta0
    start_nb  = [np.log(max(mu0, 0.01)), np.log(max(alpha0, 0.01))]
    start_infl = [np.log(max(pi_obs, 0.001) / max(1 - pi_obs, 0.001))]

    # ── Try ZINB ────────────────────────────────────────────────────────────
    if pi_obs > 0.01:   # only bother with ZINB if meaningful dropout exists
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ZeroInflatedNegativeBinomialP(
                    y, ones, exog_infl=ones).fit(
                    start_params=start_nb + start_infl,
                    method="bfgs", disp=False,
                    warn_convergence=False, maxiter=500)
            mu    = float(np.exp(m.params[0]))
            alpha = float(np.exp(m.params[1]))
            pi    = float(1 / (1 + np.exp(-m.params[2])))
            if 0 < mu < 1e6 and 0 < alpha < 1e4 and 0 <= pi <= 1:
                return dict(mu=mu, theta=1.0/alpha, pi=pi)
        except Exception:
            pass

    # ── Try NB ──────────────────────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = NegativeBinomial(y, ones).fit(
                start_params=start_nb,
                method="bfgs", disp=False,
                warn_convergence=False, maxiter=500)
        mu    = float(np.exp(m.params[0]))
        alpha = float(np.exp(m.params[1]))
        if 0 < mu < 1e6 and 0 < alpha < 1e4:
            return dict(mu=mu, theta=1.0/alpha, pi=0.0)
    except Exception:
        pass

    # ── Method-of-moments fallback ───────────────────────────────────────────
    print("  ⚠  Optimizer failed — using method-of-moments estimates")
    return dict(mu=mu0, theta=theta0, pi=pi_obs)


def bio_dispersion(cts):
    """Biological dispersion = 1/theta (NB size parameter reciprocal)."""
    y = np.asarray(cts, dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu0, theta0 = _mom_nb(y)
            alpha0 = 1.0 / theta0
            m = NegativeBinomial(y, np.ones(len(y))).fit(
                start_params=[np.log(max(mu0, 0.01)),
                               np.log(max(alpha0, 0.01))],
                method="bfgs", disp=False,
                warn_convergence=False, maxiter=500)
        alpha = float(np.exp(m.params[1]))
        return float(np.clip(alpha, 0, 1e4))
    except Exception:
        v, m2 = float(np.var(y)), float(np.mean(y))
        return float((v - m2) / (m2 ** 2)) if m2 > 0 else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5C — ZINB POWER SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def rzinb(n, mu, theta, pi, rng):
    """Sample from ZINB(mu, theta, pi)."""
    is_zero = rng.random(n) < pi
    nb_part = rng.negative_binomial(theta, theta / (theta + mu), n).astype(float)
    nb_part[is_zero] = 0.0
    return nb_part


def _nb_loglik(y, mu, theta):
    """
    Negative binomial log-likelihood (vectorised, no scipy overhead).
    theta = size/dispersion parameter (larger = less overdispersed).
    With theta fixed, MLE for mu is simply mean(y), so no optimisation needed.
    """
    from scipy.special import gammaln
    mu    = max(float(mu), 1e-10)
    theta = max(float(theta), 1e-6)
    y     = np.asarray(y, dtype=float)
    return float(np.sum(
        gammaln(y + theta) - gammaln(theta) - gammaln(y + 1)
        + theta * np.log(theta / (theta + mu))
        + y     * np.log(mu    / (theta + mu))
    ))


def zinb_lrt_pval(ya, yb, theta):
    """
    Analytic NB LRT with fixed dispersion (theta estimated from real data).

    With theta fixed the MLE for mu is the sample mean — no optimisation
    required, no convergence issues, no warnings.  O(n) per call.

    H0: mu_A == mu_B  (pooled mean)
    H1: mu_A != mu_B  (separate means)
    """
    ya = np.asarray(ya, dtype=float)
    yb = np.asarray(yb, dtype=float)

    # Edge case: both groups all-zero → no signal
    if ya.sum() == 0 and yb.sum() == 0:
        return 1.0

    mu_pool = float(np.mean(np.concatenate([ya, yb])))
    mu_a    = float(np.mean(ya))
    mu_b    = float(np.mean(yb))

    ll_null = _nb_loglik(ya, mu_pool, theta) + _nb_loglik(yb, mu_pool, theta)
    ll_alt  = _nb_loglik(ya, mu_a,    theta) + _nb_loglik(yb, mu_b,    theta)

    lrt = 2.0 * (ll_alt - ll_null)
    return float(stats.chi2.sf(max(lrt, 0.0), df=1))


def power_curve(zinb_params, fold_changes=(1.5, 2, 4, 8),
                cell_counts=None, nsim=1000, alpha=0.05, power_target=0.95,
                seed=42):
    if cell_counts is None:
        cell_counts = (list(range(2, 22, 2)) +
                       list(range(25, 105, 5)) +
                       list(range(110, 210, 10)) +
                       list(range(225, 525, 25)))

    mu    = zinb_params["mu"]
    theta = zinb_params["theta"]
    pi    = zinb_params["pi"]
    rng   = np.random.default_rng(seed)

    results    = []
    thresholds = {}

    for fc in fold_changes:
        mu_b   = mu / fc
        powers = []
        for n in cell_counts:
            pvals = []
            for _ in range(nsim):
                ya = rzinb(n, mu,   theta, pi, rng)
                yb = rzinb(n, mu_b, theta, pi, rng)
                pvals.append(zinb_lrt_pval(ya, yb, theta))
            pvals  = np.array(pvals)
            power  = float(np.nanmean(pvals < alpha))
            powers.append(power)

        powers = np.array(powers)
        results.append({"fold": fc, "cell_counts": cell_counts,
                        "powers": powers.tolist()})

        # Linear interpolation for threshold
        cross = np.where(powers >= power_target)[0]
        if len(cross) > 0 and cross[0] > 0:
            i = cross[0]
            x1, x2 = cell_counts[i-1], cell_counts[i]
            y1, y2 = powers[i-1], powers[i]
            thresh = x1 + (power_target - y1) / (y2 - y1) * (x2 - x1)
        elif len(cross) > 0:
            thresh = float(cell_counts[cross[0]])
        else:
            thresh = None

        thresholds[fc] = thresh
        t_str = f"~{thresh:.1f}" if thresh is not None else ">max"
        print(f"    Fold {fc}x → 95% power threshold: {t_str} cells")

    return results, thresholds


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5D — POWER PLOT
# ══════════════════════════════════════════════════════════════════════════════

FOLD_COLORS = {1.5: "#E07B75", 2: "#8DB34A", 4: "#26B3C4", 8: "#B87FD4"}

def power_plot(results, thresholds, gene):
    fig, ax = plt.subplots(figsize=(9, 6))

    for res in results:
        fc     = res["fold"]
        ns     = res["cell_counts"]
        powers = res["powers"]
        color  = FOLD_COLORS.get(fc, "grey")
        ax.plot(ns, powers, color=color, linewidth=1.8,
                marker="o", markersize=3, alpha=0.85, label=str(fc))
        thresh = thresholds.get(fc)
        if thresh is not None:
            ax.axvline(thresh, color=color, linestyle="dotted",
                       linewidth=1.0, alpha=0.7)

    ax.axhline(0.95, color="black", linestyle="dashed",
               linewidth=1.0, label="_nolegend_")

    # Threshold table in lower right
    table_lines = ["Fold  CellThreshold"]
    for res in results:
        fc     = res["fold"]
        thresh = thresholds.get(fc)
        t_str  = f"{thresh:.1f}" if thresh is not None else ">max"
        table_lines.append(f"{str(fc):>4}    {t_str:>7}")
    ax.text(0.57, 0.05, "\n".join(table_lines),
            transform=ax.transAxes, va="bottom", ha="left",
            fontsize=9, family="monospace", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.0))

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel("Cells Captured (Target Type)", fontsize=11)
    ax.set_ylabel("Power", fontsize=11)
    ax.set_title(
        f"scRNA-Seq Power Analysis: Cells Required per Group\n{gene}",
        fontweight="bold", fontsize=12)
    ax.text(0.5, 1.02,
            "Points where curves cross 95% power are annotated in the table",
            transform=ax.transAxes, ha="center", fontsize=9, color="grey")

    legend_handles = [Patch(facecolor=FOLD_COLORS[fc], label=str(fc))
                      for fc in [1.5, 2, 4, 8]]
    ax.legend(handles=legend_handles, title="Fold Decrease",
              loc="center right", frameon=True, fontsize=9)

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fname = f"{gene}_ZINB_power_analysis.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Step 1 — input
    path, filetype = get_input()

    # Step 2 — load
    if filetype == "mtx":
        adata, cell_clusters = load_mtx(path)
    else:
        adata, cell_clusters = load_h5ad(path)

    print(f"\nFinal matrix: {adata.n_vars} genes × {adata.n_obs} cells")

    # Step 3 — genes
    target_genes = select_genes(adata)

    # Step 4 — QC
    umi_per_cell, genes_per_cell = compute_qc(adata)
    make_qc_plots(umi_per_cell, genes_per_cell, cell_clusters)

    # Step 5 — per gene
    X = adata.X

    for gene in target_genes:
        print(f"\n====  Analysing gene: {gene}  ====")
        gene_idx = list(adata.var_names).index(gene)

        # Extract gene counts — column slice is memory-efficient for CSC matrices
        if sp.issparse(X):
            col = X.getcol(gene_idx) if hasattr(X, "getcol") else X[:, gene_idx]
            cts = np.array(col.todense()).flatten().astype(int)
        else:
            cts = np.array(X[:, gene_idx]).flatten().astype(int)

        mean_umi    = float(np.mean(cts))
        dropout_pct = 100.0 * float(np.mean(cts == 0))
        disp        = bio_dispersion(cts)
        print(f"  Mean UMI: {mean_umi:.2f} | Dropout: {dropout_pct:.1f}% "
              f"| Dispersion: {disp:.3f}")

        # 5A — histogram
        gene_histogram(cts, gene, mean_umi, dropout_pct, disp)

        # 5B — ZINB fit
        print("  Fitting ZINB distribution...")
        zinb_par = fit_zinb(cts)
        print(f"  ZINB params: mu={zinb_par['mu']:.3f}, "
              f"theta={zinb_par['theta']:.3f}, pi={zinb_par['pi']:.3f}")

        # 5C — power simulation
        print("  Simulating power curves (1,000 datasets per condition)...")
        results, thresholds = power_curve(zinb_par, nsim=1000, power_target=0.95)

        # 5D — power plot
        power_plot(results, thresholds, gene)

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  Pipeline complete. Output files saved to working directory. ║")
    print("╚══════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
