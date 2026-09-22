# CLIP Analysis for Fairness-Oriented Vision-Language Experiments

This repository contains exploratory, notebook-first work around **CLIP-based** classification and fairness analysis, with a focus on:

- prompt/label sensitivity studies on demographic predictions,
- glaucoma-related medical vision-language experiments,
- fine-tuning and linear probing workflows,
- fairness analysis (including adversarial debiasing experiments marked as work in progress).

The codebase is primarily Jupyter notebooks, with lightweight Python helpers in [`utils.py`](./utils.py) and [`gen.py`](./gen.py).

---

## Project motivation

The notebooks explore how vision-language models (especially CLIP variants) behave when predicting or analyzing sensitive attributes (e.g., race/gender context in prompt design) and medical labels (e.g., glaucoma), then evaluate distribution shifts and fairness-related behavior in the generated outputs.

Because this is research/exploration code, many experiments are iterative and notebook-driven rather than packaged into a single training CLI.

---

## Main workflows in this repository

### 1) CLIP/FairVLMed experimentation
- Notebook: [`CLIP_restuctured_version1.ipynb`](./CLIP_restuctured_version1.ipynb)
- Related notebook variant: [`CLIP_FairVLMed.ipynb`](./CLIP_FairVLMed.ipynb)

These notebooks include code for:
- dataset loading/preprocessing (`Training`, `Validation`, `Test` directory assumptions),
- CLIP inference wrappers (e.g., `RunCLIP`, `GlaucomaDataset` classes in notebook code),
- linear probing / fine-tuning style loops,
- export of prediction CSVs such as `df_test_with_preds.csv`, `df_fine_tuned_preds.csv`, and related files.

### 2) Prompt/output analysis on SCV-style CSV outputs
- Notebook: [`analyse.ipynb`](./analyse.ipynb)
- Helper module: [`utils.py`](./utils.py)
- Input directory: [`scv_1/`](./scv_1)

This workflow compares predicted vs true distributions across multiple prompt variants (race/gender-focused CSV files), and visualizes errors/confusion behavior.

### 3) Medical fairness and glaucoma analysis
- Notebook: [`analyse_glaucome.ipynb`](./analyse_glaucome.ipynb)
- Helper module: [`utils.py`](./utils.py)
- Input/output directory: [`medical/`](./medical)

This notebook loads medical metadata and prediction outputs, computes distribution/fairness-oriented summaries, and includes adversarial debiasing exploration (explicitly marked in-notebook as work in progress).

---

## Repository structure

| Path | Description |
|---|---|
| [`CLIP_restuctured_version1.ipynb`](./CLIP_restuctured_version1.ipynb) | Main end-to-end CLIP/FairVLMed-style experiment notebook (installation cells, dataset handling, inference/training/evaluation, export). |
| [`CLIP_FairVLMed.ipynb`](./CLIP_FairVLMed.ipynb) | Earlier/alternative CLIP experiment notebook with similar themes (fine-tuning and prediction export). |
| [`analyse.ipynb`](./analyse.ipynb) | Analysis notebook for CSV outputs in `scv_1/`; compares demographic prediction behavior across prompt templates. |
| [`analyse_glaucome.ipynb`](./analyse_glaucome.ipynb) | Medical/glaucoma analysis notebook using files under `medical/`, including fairness-oriented analysis utilities. |
| [`utils.py`](./utils.py) | Shared utility functions for importing CSVs and plotting (distribution comparisons, class error bars, confusion matrix). |
| [`gen.py`](./gen.py) | Small helper script that prints repetitive `analyse.ipynb` loading/stat commands for `scv_1/*.csv`. |
| [`scv_1/`](./scv_1) | CSV files used by `analyse.ipynb` (prompt variants + validation labels). |
| [`medical/`](./medical) | Medical metadata and exported experiment outputs (plus large dataset folders typically ignored by git). |

---

## Data and file expectations discovered in code

### `scv_1/` analysis inputs
- `analyse.ipynb` expects files like:
  - `./scv_1/val_labels.csv`
  - `./scv_1/r_*.csv`
  - `./scv_1/g_*.csv`
- Example discovered columns:
  - `val_labels.csv`: `file`, `age`, `gender`, `race`, `service_test`
  - prompt CSVs (e.g. `r_sans_contexte.csv`): `race`, `image`

### `medical/` analysis inputs/outputs
- `analyse_glaucome.ipynb` reads:
  - `./medical/data_summary.csv`
  - prediction exports such as `./medical/df_test_with_preds.csv`, `./medical/df_fine_tuned_preds.csv`, `./medical/df_linear_*`
  - dataset folders `./medical/Training`, `./medical/Test`, `./medical/Validation` (usually ignored in git)
- Example discovered columns:
  - `data_summary.csv`: demographic/language fields + `glaucoma`
  - `df_test_with_preds.csv`: metadata + `pred`

---

## Environment and setup

This repository currently has **no pinned dependency file** (`requirements.txt`, `environment.yml`, or `pyproject.toml` not present at root).

### Verified dependencies from imports and notebook install cells
The following packages are directly referenced in notebook code and/or `!pip install` cells:

- Core: `numpy`, `pandas`, `matplotlib`, `plotly`, `scikit-learn`, `torch`, `torchvision`, `tqdm`, `Pillow`
- CLIP / model tooling: `clip` (installed from `git+https://github.com/openai/CLIP.git` in notebook cells), `transformers`, `accelerate`, `bitsandbytes`
- Fairness / analysis: `aif360`, `fairlearn`, `umap-learn`, `seaborn`
- Data download in Colab workflows: `gdown`

### Example local setup (illustrative)
> This command block is a practical starting point inferred from notebook usage, **not an officially pinned environment**.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install numpy pandas matplotlib plotly scikit-learn torch torchvision tqdm pillow seaborn
pip install aif360 fairlearn umap-learn gdown transformers accelerate bitsandbytes
pip install git+https://github.com/openai/CLIP.git
```

If some packages are difficult to install on your platform (especially `bitsandbytes`), run only the notebook sections that do not require them.

---

## How to use

### Open notebooks
Use JupyterLab/Notebook or Google Colab:

```bash
jupyter lab
# or
jupyter notebook
```

Then open one of:
- [`CLIP_restuctured_version1.ipynb`](./CLIP_restuctured_version1.ipynb)
- [`CLIP_FairVLMed.ipynb`](./CLIP_FairVLMed.ipynb)
- [`analyse.ipynb`](./analyse.ipynb)
- [`analyse_glaucome.ipynb`](./analyse_glaucome.ipynb)

### Important path/config notes
- Some CLIP notebooks use Colab-style absolute paths like `/content/dataset` and `/content/metadata_summary.csv`.
- For local execution, update these paths to your local dataset/metadata locations before running all cells.
- `analyse.ipynb` expects `./scv_1/` CSVs in-place.
- `analyse_glaucome.ipynb` expects `./medical/` CSVs and (optionally) dataset folders.

### Optional helper script
`gen.py` can be run to print repetitive loading/stat commands used in the analysis flow:

```bash
python gen.py
```

---

## Outputs and reproducibility notes

- The repository already includes multiple exported CSV outputs under [`medical/`](./medical) and [`scv_1/`](./scv_1).
- Notebook execution can produce additional `df_*.csv` prediction/evaluation files.
- Reproducibility is currently limited by notebook-style experimentation and lack of pinned package versions / fixed run scripts.
- For reproducible reruns, consider recording:
  - exact package versions,
  - GPU/CPU environment,
  - dataset snapshot and path mapping,
  - notebook cell order and random seeds.

---

## Limitations / current status

- This is an **exploratory research repository**, not a packaged benchmark suite.
- Some notebook sections are marked as work-in-progress (e.g., adversarial debiasing exploration).
- Because setup is not fully standardized yet, expect light adaptation when running outside the original Colab-style environment.

---

## Contributing

Contributions are welcome, especially around:
- environment reproducibility (dependency pinning, setup scripts),
- notebook-to-script refactoring,
- clearer experiment tracking and evaluation reporting.

For substantial changes, opening an issue first is recommended.

