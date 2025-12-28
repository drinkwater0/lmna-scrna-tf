from __future__ import annotations

from pathlib import Path
import re
import json
import gzip
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import mmread
from scipy import sparse
from scipy.sparse import save_npz


# -----------------------------
# Study metadata (GEO / GSM)
# -----------------------------
SAMPLES = {
    "GSM8325046": ("Control", 0,  "A"),
    "GSM8325047": ("Control", 2,  "A"),
    "GSM8325048": ("Control", 4,  "A"),
    "GSM8325049": ("Control", 9,  "A"),  # Day 9A
    "GSM8325050": ("Control", 9,  "B"),  # Day 9B
    "GSM8325051": ("Control", 16, "A"),
    "GSM8325052": ("Control", 19, "A"),
    "GSM8325053": ("Control", 30, "A"),
    "GSM8325054": ("Patient", 0,  "A"),
    "GSM8325055": ("Patient", 9,  "B"),
    "GSM8325056": ("Patient", 16, "A"),
    "GSM8325057": ("Patient", 19, "A"),
}

TRAIN_SAMPLES = [
    "GSM8325046", "GSM8325047", "GSM8325048", "GSM8325050",
    "GSM8325051", "GSM8325053", "GSM8325055", "GSM8325057",
]
VAL_SAMPLES = ["GSM8325049", "GSM8325054"]
TEST_SAMPLES = ["GSM8325052", "GSM8325056"]

DAYS = [0, 2, 4, 9, 16, 19, 30]
DAY_TO_CLASS = {d: i for i, d in enumerate(DAYS)}


# -----------------------------
# Helpers
# -----------------------------
def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_unique(names: np.ndarray) -> np.ndarray:
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names.astype(str):
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
    return np.array(out, dtype=object)


def find_10x_files_flat(root: Path, which: str = "filtered") -> dict[str, dict[str, Path]]:
    root = Path(root)
    pat_mtx = re.compile(rf"(GSM\d+).*_{which}_matrix\.mtx\.gz$")
    pat_bc  = re.compile(rf"(GSM\d+).*_{which}_barcodes\.tsv\.gz$")
    pat_ft  = re.compile(rf"(GSM\d+).*_{which}_features\.tsv\.gz$")

    gsm_map: dict[str, dict[str, Path]] = {}
    for f in root.iterdir():
        if not f.is_file():
            continue
        name = f.name
        m = pat_mtx.search(name)
        if m:
            gsm_map.setdefault(m.group(1), {})["matrix"] = f
            continue
        m = pat_bc.search(name)
        if m:
            gsm_map.setdefault(m.group(1), {})["barcodes"] = f
            continue
        m = pat_ft.search(name)
        if m:
            gsm_map.setdefault(m.group(1), {})["features"] = f
            continue

    gsm_map = {gsm: d for gsm, d in gsm_map.items() if {"matrix", "barcodes", "features"} <= set(d.keys())}
    return gsm_map


def load_features(path: Path) -> np.ndarray:
    ft = pd.read_csv(path, sep="\t", header=None, compression="gzip")
    gene_symbols = ft.iloc[:, 1].astype(str).values if ft.shape[1] >= 2 else ft.iloc[:, 0].astype(str).values
    return make_unique(gene_symbols)


def load_barcodes(path: Path) -> np.ndarray:
    bc = pd.read_csv(path, sep="\t", header=None, compression="gzip")
    return bc.iloc[:, 0].astype(str).values


def load_matrix_mtx(path: Path) -> sparse.csr_matrix:
    with gzip.open(path, "rb") as fh:
        X = mmread(fh).tocsr()
    return X


def transpose_if_needed(X: sparse.csr_matrix, n_genes: int, n_cells: int) -> sparse.csr_matrix:
    # 10x mtx is typically genes x cells
    if X.shape == (n_genes, n_cells):
        return X.T.tocsr()
    if X.shape == (n_cells, n_genes):
        return X.tocsr()
    raise RuntimeError(f"Shape mismatch: X={X.shape}, genes={n_genes}, cells={n_cells}")


def downsample_rows(X: sparse.csr_matrix, barcodes: np.ndarray, max_cells: int, r: np.random.Generator):
    if max_cells <= 0 or X.shape[0] <= max_cells:
        return X, barcodes
    idx = r.choice(X.shape[0], size=max_cells, replace=False)
    idx.sort()
    return X[idx], barcodes[idx]


def qc_filter_cells(
    X: sparse.csr_matrix,
    barcodes: np.ndarray,
    mt_mask: np.ndarray,
    min_genes: int,
    max_genes: int,
    max_mito_pct: float,
):
    # n_genes_by_counts: number of nonzero genes per cell
    n_genes = X.getnnz(axis=1)
    total_counts = np.asarray(X.sum(axis=1)).ravel()

    # mito counts
    if mt_mask.any():
        X_mt = X[:, mt_mask]
        mt_counts = np.asarray(X_mt.sum(axis=1)).ravel()
    else:
        mt_counts = np.zeros_like(total_counts)

    # percent mito (avoid div by zero)
    pct_mt = np.zeros_like(total_counts, dtype=np.float32)
    nz = total_counts > 0
    pct_mt[nz] = (mt_counts[nz] / total_counts[nz]) * 100.0

    keep = (
        (n_genes >= min_genes) &
        (n_genes <= max_genes) &
        (pct_mt < max_mito_pct) &
        (total_counts > 0)
    )

    if keep.sum() == 0:
        raise RuntimeError("QC odstranilo všechny buňky. Změň prahy QC.")

    return X[keep], barcodes[keep], {
        "n_cells_before": int(X.shape[0]),
        "n_cells_after": int(keep.sum()),
        "pct_mt_mean": float(pct_mt[keep].mean()),
    }


def normalize_total_log1p_inplace(X: sparse.csr_matrix, target_sum: float = 1e4) -> sparse.csr_matrix:
    # scale each row so that row sum == target_sum, then log1p on nonzeros
    row_sums = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    scale = np.zeros_like(row_sums)
    nz = row_sums > 0
    scale[nz] = target_sum / row_sums[nz]

    # multiply rows: X = diag(scale) @ X
    X = sparse.diags(scale).dot(X).tocsr()

    # log1p on nonzeros only
    X.data = np.log1p(X.data).astype(np.float32)
    return X


def split_of(gsm: str) -> str:
    if gsm in TRAIN_SAMPLES: return "train"
    if gsm in VAL_SAMPLES:   return "val"
    if gsm in TEST_SAMPLES:  return "test"
    return "ignore"


def scale_sparse_columns_inplace(X: sparse.csr_matrix, inv_std: np.ndarray, clip: float) -> sparse.csr_matrix:
    # multiply columns by inv_std using right-multiply with diagonal matrix
    X = X.dot(sparse.diags(inv_std)).tocsr()
    if clip is not None:
        X.data = np.clip(X.data, -clip, clip).astype(np.float32)
    return X


# -----------------------------
# Main (two-pass streaming)
# -----------------------------
def main(
    raw_root: str,
    out_dir: str,
    which: str,
    seed: int,
    max_cells_per_sample: int,
    qc_min_genes: int,
    qc_max_genes: int,
    qc_max_mito_pct: float,
    min_cells_per_gene: int,
    n_hvg: int,
    scale_clip: float,
    target_sum: float,
):
    raw_root_p = Path(raw_root)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    r = rng(seed)

    gsm_files = find_10x_files_flat(raw_root_p, which=which)
    missing = sorted(set(SAMPLES.keys()) - set(gsm_files.keys()))
    if missing:
        raise RuntimeError(f"Nenašel jsem {which} soubory pro: {missing}")

    gsm_list = sorted(gsm_files.keys())

    # ---- Pass 0: define gene universe from first sample
    genes0 = load_features(gsm_files[gsm_list[0]]["features"])
    mt_mask0 = np.array([g.startswith("MT-") for g in genes0], dtype=bool)
    n_genes0 = len(genes0)

    # Stats accumulators for HVG (over normalized+log data)
    sum_g = np.zeros(n_genes0, dtype=np.float64)
    sumsq_g = np.zeros(n_genes0, dtype=np.float64)
    nnz_g = np.zeros(n_genes0, dtype=np.int64)
    n_cells_total = 0

    qc_summary = {}

    # ---- Pass 1: QC + normalize+log + accumulate gene stats (NO CONCAT)
    for gsm in tqdm(gsm_list, desc=f"PASS1 stats ({which})"):
        f = gsm_files[gsm]

        genes = load_features(f["features"])
        if len(genes) != n_genes0 or not np.all(genes == genes0):
            # If this happens, we can implement intersection mapping. For now, fail loudly.
            raise RuntimeError(f"[{gsm}] features.tsv.gz se liší od prvního vzorku (jiné geny/pořadí).")

        barcodes = load_barcodes(f["barcodes"])
        X = load_matrix_mtx(f["matrix"])
        X = transpose_if_needed(X, n_genes=n_genes0, n_cells=len(barcodes))

        X, barcodes = downsample_rows(X, barcodes, max_cells_per_sample, r)
        X, barcodes, qcinfo = qc_filter_cells(
            X, barcodes, mt_mask0, qc_min_genes, qc_max_genes, qc_max_mito_pct
        )
        qc_summary[gsm] = qcinfo

        X = normalize_total_log1p_inplace(X, target_sum=target_sum)

        # accumulate stats
        sum_g += np.asarray(X.sum(axis=0)).ravel()
        sumsq_g += np.asarray(X.power(2).sum(axis=0)).ravel()
        nnz_g += X.getnnz(axis=0)
        n_cells_total += X.shape[0]

    # gene eligibility: expressed in at least min_cells_per_gene cells
    eligible = nnz_g >= int(min_cells_per_gene)
    if eligible.sum() == 0:
        raise RuntimeError("Po gene filtru nezbyl žádný gen. Sniž min_cells_per_gene.")

    mean = sum_g / float(n_cells_total)
    ex2 = sumsq_g / float(n_cells_total)
    var = ex2 - mean * mean
    var[var < 0] = 0.0

    # pick HVG among eligible
    eligible_idx = np.where(eligible)[0]
    var_eligible = var[eligible_idx]
    topk = min(int(n_hvg), len(eligible_idx))
    top_rel = np.argpartition(-var_eligible, kth=topk - 1)[:topk]
    hvg_idx = np.sort(eligible_idx[top_rel])

    genes_hvg = genes0[hvg_idx]

    # std for scaling (zero_center=False): divide by std
    std = np.sqrt(var[hvg_idx]).astype(np.float32)
    std[std == 0] = 1.0
    inv_std = (1.0 / std).astype(np.float32)

    # ---- Pass 2: build final sparse X (only HVG) + labels
    X_blocks = []
    y_geno_all = []
    y_day_all = []
    gsm_all = []
    split_all = []
    cell_id_all = []

    for gsm in tqdm(gsm_list, desc=f"PASS2 build ({which})"):
        f = gsm_files[gsm]
        barcodes = load_barcodes(f["barcodes"])
        X = load_matrix_mtx(f["matrix"])
        X = transpose_if_needed(X, n_genes=n_genes0, n_cells=len(barcodes))

        X, barcodes = downsample_rows(X, barcodes, max_cells_per_sample, r)
        X, barcodes, _ = qc_filter_cells(
            X, barcodes, mt_mask0, qc_min_genes, qc_max_genes, qc_max_mito_pct
        )
        X = normalize_total_log1p_inplace(X, target_sum=target_sum)

        # subset HVG
        X = X[:, hvg_idx].tocsr()

        # scale (no centering) + clip
        X = scale_sparse_columns_inplace(X, inv_std=inv_std, clip=scale_clip)

        sp = split_of(gsm)
        if sp == "ignore":
            continue

        geno, day, rep = SAMPLES[gsm]
        y_geno = 1 if geno == "Patient" else 0
        y_day = DAY_TO_CLASS[day]

        n = X.shape[0]
        X_blocks.append(X)
        y_geno_all.append(np.full(n, y_geno, dtype=np.int64))
        y_day_all.append(np.full(n, y_day, dtype=np.int64))
        gsm_all.append(np.full(n, gsm, dtype=object))
        split_all.append(np.full(n, sp, dtype=object))
        # make unique cell ids
        cell_id_all.append(np.array([f"{gsm}:{bc}" for bc in barcodes], dtype=object))

    if not X_blocks:
        raise RuntimeError("Nevznikla žádná data (X_blocks je prázdné).")

    X_final = sparse.vstack(X_blocks, format="csr").astype(np.float32)
    y_geno_final = np.concatenate(y_geno_all)
    y_day_final = np.concatenate(y_day_all)
    gsm_final = np.concatenate(gsm_all)
    split_final = np.concatenate(split_all)
    cell_id_final = np.concatenate(cell_id_all)

    # ---- Save
    save_npz(out_dir_p / "X_csr.npz", X_final)
    np.save(out_dir_p / "y_geno.npy", y_geno_final)
    np.save(out_dir_p / "y_day.npy", y_day_final)
    np.save(out_dir_p / "gsm.npy", gsm_final)
    np.save(out_dir_p / "split.npy", split_final)
    np.save(out_dir_p / "cell_id.npy", cell_id_final)
    np.save(out_dir_p / "genes.npy", genes_hvg)

    meta = {
        "source": "GSE269705",
        "which": which,
        "seed": seed,
        "target_sum": float(target_sum),
        "qc": {
            "min_genes": int(qc_min_genes),
            "max_genes": int(qc_max_genes),
            "max_mito_pct": float(qc_max_mito_pct),
            "min_cells_per_gene": int(min_cells_per_gene),
        },
        "hvg": {
            "n_hvg": int(len(hvg_idx)),
            "method": "variance_on_log_normalized (streaming)",
        },
        "scale": {"zero_center": False, "clip": float(scale_clip)},
        "splits": {
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
            "test_samples": TEST_SAMPLES,
        },
        "day_to_class": DAY_TO_CLASS,
        "pass1": {
            "n_cells_total_after_qc": int(n_cells_total),
            "eligible_genes": int(eligible.sum()),
        },
        "final": {
            "n_cells": int(X_final.shape[0]),
            "n_genes": int(X_final.shape[1]),
            "nnz": int(X_final.nnz),
            "label_counts": {
                "patient_cells": int((y_geno_final == 1).sum()),
                "control_cells": int((y_geno_final == 0).sum()),
                "split": {s: int((split_final == s).sum()) for s in ["train", "val", "test"]},
            },
        },
        "qc_per_sample": qc_summary,
    }
    (out_dir_p / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("✅ Saved:")
    print("  ", out_dir_p / "X_csr.npz")
    print("  ", out_dir_p / "y_geno.npy, y_day.npy, gsm.npy, split.npy, cell_id.npy, genes.npy")
    print("  ", out_dir_p / "meta.json")
    print(f"  Final: cells={X_final.shape[0]} genes={X_final.shape[1]} nnz={X_final.nnz}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default="data/raw/extracted")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--which", choices=["filtered", "raw"], default="filtered")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_cells_per_sample", type=int, default=3000)  # start safe

    ap.add_argument("--qc_min_genes", type=int, default=200)
    ap.add_argument("--qc_max_genes", type=int, default=6000)
    ap.add_argument("--qc_max_mito_pct", type=float, default=20.0)

    ap.add_argument("--min_cells_per_gene", type=int, default=10)
    ap.add_argument("--n_hvg", type=int, default=1000)

    ap.add_argument("--scale_clip", type=float, default=10.0)
    ap.add_argument("--target_sum", type=float, default=1e4)

    args = ap.parse_args()

    main(
        raw_root=args.raw_root,
        out_dir=args.out_dir,
        which=args.which,
        seed=args.seed,
        max_cells_per_sample=args.max_cells_per_sample,
        qc_min_genes=args.qc_min_genes,
        qc_max_genes=args.qc_max_genes,
        qc_max_mito_pct=args.qc_max_mito_pct,
        min_cells_per_gene=args.min_cells_per_gene,
        n_hvg=args.n_hvg,
        scale_clip=args.scale_clip,
        target_sum=args.target_sum,
    )

