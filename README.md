# Explainable CNN–BiLSTM–Attention Autoencoder for Unsupervised Network Intrusion Detection

Code for the paper *"An Explainable CNN–BiLSTM–Attention Autoencoder for Unsupervised Network Intrusion Detection:
A Leakage-Controlled Evaluation on Three Benchmarks"* (B. J. Naik and S. Kumar, NIT Kurukshetra).

One notebook produces every number, table and figure in the paper, for one dataset per run.

## Files

| File | Purpose |
|---|---|
| `ids_revision.py` | The full experiment (same code as the notebook, as a plain script). |
| `ids_revision_notebook.ipynb` | The notebook as run on Kaggle. Set `DATASET` in the first code cell. |
| `make_tables.py` | Turns the `results_<DATASET>/` folders into the LaTeX tables, figures and number macros of the paper. |
| `make_diagrams.py` | Draws the architecture and protocol figures. |

## Data

All datasets are public. Attach them to the Kaggle notebook (or put them under one folder and set `IDS_INPUT_ROOT`):

| Dataset | Source used | Files |
|---|---|---|
| CIC-DDoS2019 | Kaggle `dhoogla/cicddos2019` (curated parquet release) | `*-training.parquet`, `*-testing.parquet` |
| CICIDS2017 | MachineLearningCVE CSV release | `*.pcap_ISCX.csv` (8 files) |
| UNSW-NB15 | Official partitioned sets | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` |

## Running

On Kaggle: attach the dataset, enable a GPU, set `DATASET` to `"CICDDOS2019"`, `"CICIDS2017"` or `"UNSWNB15"`, and run
all cells. A full run (15 models × 5 seeds plus explanations and sensitivity analyses) takes about 1.5–3 hours on
2× T4. Set the environment variable `IDS_QUICK=1` for a 2-minute smoke test.

Locally:

```bash
IDS_DATASET=CICIDS2017 IDS_INPUT_ROOT=/path/to/data IDS_OUT_ROOT=./out python ids_revision.py
python make_tables.py ./out
```

Environment used for the paper (Kaggle): Python 3.13, TensorFlow 2.20, Keras 3.13, scikit-learn 1.6, NumPy 2.1,
pandas 2.3, 2× NVIDIA T4.

## Protocol in brief

* Training and test data come from different capture days or official partitions
  (CIC-DDoS2019: training-day vs testing-day files; CICIDS2017: Mon–Wed vs Thu–Fri; UNSW-NB15: official split).
* Unsupervised detectors see benign flows of the training days only. Benign rows are assigned to fit (72 %),
  early-stopping validation (8 %) and threshold calibration (20 %) in random contiguous 200-row blocks.
* Threshold θ = μ + 2σ of the anomaly scores of the benign calibration windows; no attack label is used for any choice.
* Windows of T = 20 flows; test windows are non-overlapping and cover the full mixed test stream; a window is an
  attack if it contains at least one attack flow.
* Every model uses the same windows, calibration rule and seeds 0–4.

## Outputs (`results_<DATASET>/`)

`runs.csv` (every model × seed, all metrics), `summary_mean_sd.csv`, `paired_tests.csv`, `per_category_recall.csv`,
`threshold_and_label_sensitivity.csv`, `sensitivity.csv` (window length, bottleneck size, shortcut features),
`latency.csv`, `explain_*.csv/json` (deletion tests, stability, attention statistics), `example_explanation_report.txt`,
`test_scores.npz` (all test scores, so any metric can be recomputed), `data_summary.json`, `config.json`, and `figures/`.

## Citation

Please cite the paper if you use this code. *(BibTeX to be added on acceptance.)*

## License

*(Choose a license before publishing, e.g. MIT.)*
