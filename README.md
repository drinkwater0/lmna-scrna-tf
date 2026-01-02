


Original - split is mix of days ->
    Train has 8 GSM samples
    Val has (1 control, 1 patient) both from different days
    Test has (1 control, 1 patient) both from different days
-> day–genotype confounding: day/timepoint strongly affects expression, and the split makes some days appear with only one genotype in training, so the model can learn a spurious shortcut “day-associated expression pattern → genotype”. When the day↔genotype association differs between train and test, this shortcut can systematically flip predictions, producing an inverted ROC curve (AUC < 0.5).

split_scheme - split by days intact ->
    Test has 2 samples from the same day
    Val has 2 samples from the same day
    Train all of the left samples
-> days are disjoint across train/val/test, and genotype is evaluated within the same day in val/test, so day-specific expression patterns alone cannot solve the task.

run A/B models 
- Model A -> single-task: predicts genotype from RNA expression vectors.

- Model B -> multitask: same RNA vectors as input, but two output heads:
    (1) genotype (binary)
    (2) day/timepoint (multiclass, auxiliary target)
    Day is NOT an input feature.



python src/prepare_dataset.py --out_dir data/processed_original --split_scheme original
python src/prepare_dataset.py --out_dir data/processed_dm_t0_v16 --split_scheme day_matched --split_test_day 0  --split_val_day 16
python src/prepare_dataset.py --out_dir data/processed_dm_t16_v0 --split_scheme day_matched --split_test_day 16 --split_val_day 0
python src/prepare_dataset.py --out_dir data/processed_dm_t19_v0 --split_scheme day_matched --split_test_day 19 --split_val_day 0


python src/train.py --data_dir data/processed_original --out_dir outputs/runs/orig_A --model A
python src/train.py --data_dir data/processed_original --out_dir outputs/runs/orig_B --model B

python src/train.py --data_dir data/processed_dm_t0_v16 --out_dir outputs/runs/dm_t0_v16_A --model A 
python src/train.py --data_dir data/processed_dm_t0_v16 --out_dir outputs/runs/dm_t0_v16_B --model B 
python src/train.py --data_dir data/processed_dm_t16_v0 --out_dir outputs/runs/dm_t16_v0_A --model A
python src/train.py --data_dir data/processed_dm_t16_v0 --out_dir outputs/runs/dm_t16_v0_B --model B
python src/train.py --data_dir data/processed_dm_t19_v0 --out_dir outputs/runs/dm_t19_v0_A --model A
python src/train.py --data_dir data/processed_dm_t19_v0 --out_dir outputs/runs/dm_t19_v0_B --model B


python src/evaluate.py --run_dir outputs/runs/orig_A 
python src/evaluate.py --run_dir outputs/runs/orig_B 
python src/evaluate.py --run_dir outputs/runs/dm_t0_v16_A 
python src/evaluate.py --run_dir outputs/runs/dm_t0_v16_B
python src/evaluate.py --run_dir outputs/runs/dm_t16_v0_A 
python src/evaluate.py --run_dir outputs/runs/dm_t16_v0_B
python src/evaluate.py --run_dir outputs/runs/dm_t19_v0_A
python src/evaluate.py --run_dir outputs/runs/dm_t19_v0_B 


LMNA scRNA-seq TensorFlow pipeline

This repo contains a fully repeatable pipeline for dataset preparation, training, sweeps, and evaluation.

Setup
1) Create the environment (recommended)
   conda env create -f environment.yml
   conda activate lmna-scrna-tf

2) If you do not use conda, install dependencies listed in environment.yml
   (TensorFlow 2.15.* expects Python 3.11).

Data download and extraction
1) Download GEO raw archive (about 4.8 GB)
   python src/download_geo.py

2) Extract files (expects flat directory with *_matrix.mtx.gz, *_barcodes.tsv.gz, *_features.tsv.gz)
   mkdir -p data/raw/extracted
   tar -xf data/raw/GSE269705_RAW.tar -C data/raw/extracted

Prepare datasets (reproducible splits)
Default behavior uses filtered 10x files.

Original split (baseline)
  python src/prepare_dataset.py --out_dir data/processed_original --split_scheme original

Day-matched split (recommended for leakage control)
  python src/prepare_dataset.py --out_dir data/processed_dm_t19_v0 --split_scheme day_matched --split_test_day 19 --split_val_day 0

You can create multiple day-matched variants as needed:
  python src/prepare_dataset.py --out_dir data/processed_dm_t0_v16 --split_scheme day_matched --split_test_day 0  --split_val_day 16
  python src/prepare_dataset.py --out_dir data/processed_dm_t16_v0 --split_scheme day_matched --split_test_day 16 --split_val_day 0

Minimal course sweep (Model A only)
This sweep covers:
single-layer vs multi-layer, epochs, batch size, activation, optimizer.

  python src/sweep.py --model A --data_dir data/processed_dm_t19_v0

The results CSV is written to:
  outputs/runs/results.csv

You can choose the CSV name or path:
  python src/sweep.py --model A --data_dir data/processed_dm_t19_v0 --out_csv results_dm_t19.csv

Model B (auxiliary day head)
Model B uses the best Model A config and varies only loss_w_day (0.1/0.3/0.5).
It picks the best A run from outputs/runs/results.csv unless you override it.

  python src/sweep.py --model B --data_dir data/processed_dm_t19_v0

Optional overrides:
  python src/sweep.py --model B --best_a_run A_d2_e20_b128_relu_adam --data_dir data/processed_dm_t19_v0
  python src/sweep.py --model B --best_a_csv outputs/runs/results_A.csv --data_dir data/processed_dm_t19_v0

Evaluation plots
After any training run, generate plots and a summary:
  python src/evaluate.py --run_dir outputs/runs/<run_name>

Artifacts written per run
- outputs/runs/<run_name>/model.keras
- outputs/runs/<run_name>/history.json
- outputs/runs/<run_name>/config.json
- outputs/runs/<run_name>/test_metrics.json
- outputs/runs/<run_name>/plots/*.png
- outputs/runs/<run_name>/y_test.npy, p_test.npy (and day outputs for Model B)

Reproducibility notes
- The dataset preparation is seeded; keep the same split flags for repeatable splits.
- The training script uses a fixed seed; results should be repeatable when the same
  environment and GPU/CPU backend are used.