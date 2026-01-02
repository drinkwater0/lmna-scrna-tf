# LMNA scRNA-seq classification (TensorFlow/Keras)

Reproducible pipeline to classify **patient vs control status** from single-cell RNA-seq (scRNA-seq) in GEO **GSE269705**.
The repository implements:

- RAM-safe preprocessing of 10x-style matrices into an ML-ready sparse feature matrix
- **GSM-level** train/val/test splits (no within-library leakage)
- A **day-matched** split scheme to reduce day-genotype confounding
- Two models:
  - **Model A**: genotype-only MLP
  - **Model B**: multitask MLP (genotype + auxiliary day head)
- Training, evaluation plots, and a small sweep (course requirement)

If you use this repo for the paper, include the compiled PDF in `paper/` and link it from here.

---

## Repository structure

- `src/download_geo.py` - download GEO raw archive
- `src/prepare_dataset.py` - preprocessing + split creation (train-only HVG/scaling)
- `src/train.py` - train Model A or B on a prepared dataset
- `src/evaluate.py` - generate ROC/curves/confusion matrix and summary JSON
- `src/sweep.py` - small sweep for Model A and a simple Model B follow-up

Typical outputs go to:
- `data/processed_*` - prepared datasets (sparse matrices + labels + metadata)
- `outputs/runs/<run_name>/` - models, metrics, plots

---

## Setup

### Option A: conda (recommended)
```bash
conda env create -f environment.yml
conda activate lmna-scrna-tf
```

### Option B: pip
Install dependencies equivalent to `environment.yml`.
Note: TensorFlow 2.15.* expects Python 3.11.

---

## Data download + extraction (GSE269705)

1) Download the raw GEO archive (~4.8 GB):
```bash
python src/download_geo.py
```

2) Extract into a flat directory:
```bash
mkdir -p data/raw/extracted
tar -xf data/raw/GSE269705_RAW.tar -C data/raw/extracted
```

Expected per-GSM files:
- `*_matrix.mtx.gz`
- `*_barcodes.tsv.gz`
- `*_features.tsv.gz`

---

## Dataset preparation (reproducible splits)

All splits are performed at the **GSM (library) level**.

### Original split (baseline)
```bash
python src/prepare_dataset.py \
  --out_dir data/processed_original \
  --split_scheme original
```

### Day-matched split (confound-reduced; recommended)
Choose `--split_test_day` and `--split_val_day`. Example used in the paper:
```bash
python src/prepare_dataset.py \
  --out_dir data/processed_dm_t19_v0 \
  --split_scheme day_matched \
  --split_test_day 19 \
  --split_val_day 0
```

Other variants (also used in comparisons):
```bash
python src/prepare_dataset.py --out_dir data/processed_dm_t0_v16  --split_scheme day_matched --split_test_day 0  --split_val_day 16
python src/prepare_dataset.py --out_dir data/processed_dm_t16_v0  --split_scheme day_matched --split_test_day 16 --split_val_day 0
```

### Leakage control (important)
**HVG selection and scaling are fit on TRAIN only**, then frozen and applied to validation/test.
This prevents preprocessing leakage.

---

## Training (Model A / Model B)

### Train Model A (genotype-only)
```bash
python src/train.py \
  --data_dir data/processed_dm_t19_v0 \
  --out_dir outputs/runs/dm_t19_v0_A \
  --model A
```

### Train Model B (genotype + auxiliary day head)
```bash
python src/train.py \
  --data_dir data/processed_dm_t19_v0 \
  --out_dir outputs/runs/dm_t19_v0_B \
  --model B
```

---

## Evaluation plots

After any run:
```bash
python src/evaluate.py --run_dir outputs/runs/<run_name>
```

This writes:
- `plots/*.png` (ROC curve, confusion matrix, training curves, etc.)
- `plots_summary.json` (plot summary for that run)
- `y_test.npy`, `p_test.npy` (and day outputs for Model B)

---

## Minimal sweep (course requirement)

### Model A sweep (depth / epochs / batch / activation / optimizer)
```bash
python src/sweep.py --model A --data_dir data/processed_dm_t19_v0
```

By default it writes:
- `outputs/runs/results.csv`

Custom CSV:
```bash
python src/sweep.py --model A --data_dir data/processed_dm_t19_v0 --out_csv results_dm_t19.csv
```

### Model B sweep (loss weight for auxiliary day head)
Model B uses the best Model A config and varies `loss_w_day` (e.g., 0.1/0.3/0.5).
It selects the best A run from the results CSV unless overridden.

```bash
python src/sweep.py --model B --data_dir data/processed_dm_t19_v0
```

Overrides:
```bash
python src/sweep.py --model B --best_a_run A_d2_e20_b128_relu_adam --data_dir data/processed_dm_t19_v0
python src/sweep.py --model B --best_a_csv outputs/runs/results_A.csv --data_dir data/processed_dm_t19_v0
```

---

## Full reproduction (paper tables/figures)

Run this from the repo root after setting up the environment and extracting data:

```bash
# 1) Build datasets (train-only HVG/scaling)
python src/prepare_dataset.py --out_dir data/processed_original  --split_scheme original
python src/prepare_dataset.py --out_dir data/processed_dm_t0_v16 --split_scheme day_matched --split_test_day 0  --split_val_day 16
python src/prepare_dataset.py --out_dir data/processed_dm_t16_v0 --split_scheme day_matched --split_test_day 16 --split_val_day 0
python src/prepare_dataset.py --out_dir data/processed_dm_t19_v0 --split_scheme day_matched --split_test_day 19 --split_val_day 0

# 2) Train default runs for Table 1 (A+B on each split)
python src/train.py --data_dir data/processed_original  --out_dir outputs/runs/orig_A      --model A
python src/train.py --data_dir data/processed_original  --out_dir outputs/runs/orig_B      --model B
python src/train.py --data_dir data/processed_dm_t0_v16 --out_dir outputs/runs/dm_t0_v16_A --model A
python src/train.py --data_dir data/processed_dm_t0_v16 --out_dir outputs/runs/dm_t0_v16_B --model B
python src/train.py --data_dir data/processed_dm_t16_v0 --out_dir outputs/runs/dm_t16_v0_A --model A
python src/train.py --data_dir data/processed_dm_t16_v0 --out_dir outputs/runs/dm_t16_v0_B --model B
python src/train.py --data_dir data/processed_dm_t19_v0 --out_dir outputs/runs/dm_t19_v0_A --model A
python src/train.py --data_dir data/processed_dm_t19_v0 --out_dir outputs/runs/dm_t19_v0_B --model B

# 3) Evaluation plots for each run (optional, but recommended)
python src/evaluate.py --run_dir outputs/runs/orig_A
python src/evaluate.py --run_dir outputs/runs/orig_B
python src/evaluate.py --run_dir outputs/runs/dm_t0_v16_A
python src/evaluate.py --run_dir outputs/runs/dm_t0_v16_B
python src/evaluate.py --run_dir outputs/runs/dm_t16_v0_A
python src/evaluate.py --run_dir outputs/runs/dm_t16_v0_B
python src/evaluate.py --run_dir outputs/runs/dm_t19_v0_A
python src/evaluate.py --run_dir outputs/runs/dm_t19_v0_B

# 4) Sweep table (Model A on dm_t19_v0)
python src/sweep.py --model A --data_dir data/processed_dm_t19_v0 --out_csv results.csv
```

If your repo includes helper scripts for the paper:
```bash
python src/make_table1.py
python src/plot_auc_splits.py
```

---

## Notes on reproducibility

- Dataset preparation is seeded; using the same split flags yields repeatable splits.
- Training also uses a fixed seed; minor numeric differences may still occur across CPU/GPU backends.

---

## Data


- Data: GSE269705 is hosted by NCBI GEO; this repo provides scripts to download and process it.
