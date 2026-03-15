# DrugSol

<p align="center">
  <img src="docs/images/drugsol_logo.png" alt="DrugSol Logo" width="200"/>
</p>

<p align="center">
  <strong>Machine Learning Pipeline for Aqueous Solubility Prediction</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#pipeline-overview">Pipeline Overview</a> •
  <a href="#documentation">Documentation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Nextflow-DSL2-green?logo=nextflow" alt="Nextflow DSL2"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License MIT"/>
  <img src="https://img.shields.io/badge/Version-1.0.0-orange" alt="Version 1.0.0"/>
</p>

---

## Overview

**DrugSol** is an end-to-end machine learning pipeline for predicting aqueous solubility (logS) of drug-like compounds. Built with Nextflow DSL2, it provides a reproducible, scalable, and production-ready workflow for pharmaceutical research and drug discovery.

The pipeline implements a state-of-the-art ensemble approach combining:
- **Gradient Boosting Models**: XGBoost, LightGBM, CatBoost
- **Graph Neural Networks**: Chemprop D-MPNN
- **Physics-informed Baseline**: Ridge regression with thermodynamic features

### Why Solubility Matters

Aqueous solubility is a critical physicochemical property in drug development:
- **~40% of drug candidates fail** due to poor solubility
- Directly impacts **bioavailability** and **absorption**
- Essential for **formulation development**
- Required by regulatory agencies (FDA, EMA)

---

## Features

### 🔬 Scientific Features
- **Multi-source data integration**: BigSolDB, ChEMBL, custom datasets
- **Automated data curation**: Water solvent filtering, temperature range selection, outlier detection
- **SMILES standardization**: Neutralization, tautomer canonicalization, salt removal
- **Dual feature engineering**: 1,600+ Mordred descriptors + RDKit physicochemical properties
- **ChemBERTa embeddings**: Transformer-based molecular representations
- **pH-dependent corrections**: Henderson-Hasselbalch thermodynamic adjustments

### 🛠️ Technical Features
- **Nextflow DSL2**: Modular, reproducible workflows
- **Conda environments**: Automatic dependency management
- **GPU acceleration**: CUDA support for Chemprop and GBM training
- **Cross-validation**: Stratified K-fold with Optuna hyperparameter tuning
- **Ensemble learning**: Stacking and blending meta-learners
- **Two operational modes**: Research (training) and Execution (inference)

---

## Installation

### Prerequisites

- **Nextflow** ≥ 22.10.1
- **Micromamba** or Conda
- **Python** 3.8+ (managed by Conda)
- **CUDA** 11.x (optional, for GPU acceleration)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/drugsol.git
cd drugsol

# 2. Install Nextflow (if not already installed)
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/

# 3. Install Micromamba (recommended over Conda)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# 4. Verify installation
nextflow -version
micromamba --version
```

### First Run

The pipeline automatically creates Conda environments on first execution:

```bash
nextflow run main.nf --mode research -profile gpu_small --n_iterations 1
```

---

## Quick Start

### Research Mode (Training)

Train models with cross-validation on public datasets:

```bash
# Full training pipeline (10 iterations, 5-fold CV)
nextflow run main.nf --mode research -profile gpu_small

# Quick test (1 iteration)
nextflow run main.nf --mode research -profile gpu_small --n_iterations 1

# CPU-only execution
nextflow run main.nf --mode research -profile standard
```

### Execution Mode (Inference)

Predict solubility for new molecules:

```bash
# Using trained models from research phase
nextflow run main.nf --mode execution --input molecules.csv -profile standard

# With specific model override
nextflow run main.nf --mode execution --input molecules.csv --model /path/to/model
```

### Input Format

For execution mode, provide a CSV/TSV/Parquet file with SMILES:

```csv
smiles,name
CC(=O)OC1=CC=CC=C1C(=O)O,Aspirin
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,Caffeine
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,Ibuprofen
```

---

## Pipeline Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DrugSol Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   INGEST     │───▶│    CURATE    │───▶│   PREPARE    │                   │
│  │              │    │              │    │              │                   │
│  │ • BigSolDB   │    │ • Filter H2O │    │ • Mordred    │                   │
│  │ • ChEMBL     │    │ • Temp range │    │ • RDKit      │                   │
│  │ • Custom     │    │ • Outliers   │    │ • ChemBERTa  │                   │
│  │              │    │ • SMILES std │    │ • Folds      │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                 │                            │
│                                                 ▼                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         TRAIN (OOF)                                   │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │   │
│  │  │ XGBoost │  │ LightGBM│  │ CatBoost│  │ Chemprop│  │ Physics │     │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘     │   │
│  │       │            │            │            │            │           │   │
│  │       └────────────┴────────────┴────────────┴────────────┘           │   │
│  │                              │                                        │   │
│  │                              ▼                                        │   │
│  │                    ┌─────────────────┐                                │   │
│  │                    │  Meta-Learner   │                                │   │
│  │                    │  (Stack/Blend)  │                                │   │
│  │                    └─────────────────┘                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                 │                            │
│                                                 ▼                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  PRODUCTION  │───▶│   EVALUATE   │───▶│   PUBLISH    │                   │
│  │              │    │              │    │              │                   │
│  │ • Full train │    │ • Metrics    │    │ • Model card │                   │
│  │ • Ensemble   │    │ • Plots      │    │ • Resources  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Operational Modes

| Mode | Purpose | Input | Output |
|------|---------|-------|--------|
| **Research** | Train and validate models | Public databases | Trained ensemble + metrics |
| **Execution** | Predict new compounds | SMILES file | Solubility predictions |

---

## Project Structure

```
drugsol/
├── main.nf                     # Pipeline entrypoint
├── nextflow.config             # Global configuration
│
├── subworkflows/
│   └── modes/
│       ├── research/
│       │   └── research.nf     # Training workflow
│       └── execution/
│           └── execution.nf    # Inference workflow
│
├── modules/                    # Nextflow process modules
│   ├── fetch_bigsoldb/         # Data ingestion
│   ├── fetch_chembl/
│   ├── filter_water/           # Data curation
│   ├── filter_by_temperature_range/
│   ├── detect_outliers/
│   ├── standardize_smiles/
│   ├── make_features_mordred/  # Feature engineering
│   ├── make_features_rdkit/
│   ├── make_embeddings_chemberta/
│   ├── train_oof_gbm/          # Model training
│   ├── train_oof_chemprop/
│   ├── train_oof_physics/
│   ├── meta_stack_blend/       # Ensemble learning
│   ├── final_report/           # Evaluation
│   └── ...
│
├── bin/                        # Python scripts
│   ├── fetch_bigsoldb.py
│   ├── standardize_smiles.py
│   ├── make_features_mordred.py
│   ├── train_oof_gbm.py
│   ├── train_oof_chemprop.py
│   └── ...
│
├── envs/                       # Conda environments
│   ├── drugsol-data.yml        # Data processing
│   ├── drugsol-train.yml       # Model training
│   └── drugsol-bert.yml        # ChemBERTa
│
├── resources/                  # Reference files
│   ├── smarts_pattern_ionized.txt
│   └── ...
│
└── results/                    # Pipeline outputs
    ├── research/
    │   ├── ingest/
    │   ├── curate/
    │   ├── prepare_data/
    │   ├── training/
    │   ├── final_product/
    │   └── pipeline_info/
    └── execution/
        └── predictions/
```

---

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `research` | Pipeline mode: `research` or `execution` |
| `--input` | `null` | Input file for execution mode |
| `--outdir` | `results/<mode>` | Output directory |
| `--n_iterations` | `10` | Number of CV iterations |
| `--n_cv_folds` | `5` | Number of CV folds |
| `--random_seed` | `42` | Random seed for reproducibility |

### Profiles

| Profile | Use Case | GPU | Memory |
|---------|----------|-----|--------|
| `standard` | CPU-only, testing | ❌ | Low |
| `gpu_small` | Consumer GPU (RTX 3070) | ✅ | 6-8 GB |
| `gpu_high` | Workstation (A5000/A6000) | ✅ | 32+ GB |

### Example Configurations

```bash
# High-performance training
nextflow run main.nf \
    --mode research \
    --n_iterations 20 \
    --n_cv_folds 10 \
    -profile gpu_high

# Skip specific models
nextflow run main.nf \
    --mode research \
    --skip_chemprop true \
    --skip_catboost true \
    -profile standard

# Custom temperature range
nextflow run main.nf \
    --mode research \
    --temp_min_celsius 20 \
    --temp_max_celsius 40 \
    -profile gpu_small
```

---

## Models

### Base Models

| Model | Type | Features | Hyperparameter Tuning |
|-------|------|----------|----------------------|
| **XGBoost** | Gradient Boosting | Mordred + ChemBERTa | Optuna (50 trials) |
| **LightGBM** | Gradient Boosting | Mordred + ChemBERTa | Optuna (50 trials) |
| **CatBoost** | Gradient Boosting | Mordred + ChemBERTa | Optuna (50 trials) |
| **Chemprop** | D-MPNN (GNN) | SMILES only | Optuna (20 trials) |
| **Physics** | Ridge Regression | RDKit + Engineered | GridSearchCV |

### Ensemble Strategy

The meta-learner combines base model predictions using:
1. **Stacking**: Ridge regression on OOF predictions
2. **Blending**: Weighted average based on validation performance

---

## Output

### Research Mode

```
results/research/
├── ingest/
│   ├── bigsoldb.csv
│   └── chembl_solubility.csv
├── curate/
│   ├── filtered_water.parquet
│   ├── filtered_temperature.parquet
│   └── standardized_smiles.parquet
├── prepare_data/
│   ├── iter_1/
│   │   ├── train_features_mordred.parquet
│   │   ├── train_chemberta_embeddings.parquet
│   │   └── folds.parquet
│   └── ...
├── training/
│   ├── iter_1/
│   │   ├── oof_gbm/
│   │   ├── oof_gnn/
│   │   └── oof_physics/
│   └── ...
├── final_product/
│   ├── drugsol_model/
│   │   ├── model_card.json
│   │   ├── xgboost_final.pkl
│   │   ├── lightgbm_final.pkl
│   │   ├── catboost_final.cbm
│   │   ├── chemprop_final/
│   │   └── meta_weights.json
│   └── final_report.html
└── pipeline_info/
    ├── execution_timeline.html
    └── execution_report.html
```

### Execution Mode

```
results/execution/
└── predictions/
    ├── predictions_raw.csv
    └── predictions_physio_pH7.4.csv
```

---

## Performance

### Expected Metrics (BigSolDB + ChEMBL)

| Model | RMSE (logS) | R² | MAE |
|-------|-------------|-----|-----|
| XGBoost | ~0.85 | ~0.82 | ~0.62 |
| LightGBM | ~0.84 | ~0.83 | ~0.61 |
| CatBoost | ~0.86 | ~0.81 | ~0.63 |
| Chemprop | ~0.92 | ~0.78 | ~0.68 |
| Physics | ~1.10 | ~0.70 | ~0.82 |
| **Ensemble** | **~0.80** | **~0.85** | **~0.58** |

### Runtime (GPU, 1 iteration)

| Stage | Time |
|-------|------|
| Ingest + Curate | ~5 min |
| Feature Engineering | ~15 min |
| GBM Training (3 models) | ~30 min |
| Chemprop Training | ~45 min |
| Full Training + Ensemble | ~20 min |
| **Total** | **~2 hours** |

---

## Troubleshooting

### Common Issues

#### Conda Environment Failures

```bash
# Reset environments
rm -rf envs/conda_cache/drugsol-*
rm -rf .nextflow
nextflow run main.nf --mode research -profile gpu_small
```

#### Out of Memory (GPU)

```bash
# Use smaller batches
nextflow run main.nf \
    --mode research \
    --chemprop_batch_size 16 \
    --gbm_tune_trials 20 \
    -profile gpu_small
```

#### Missing Dependencies

```bash
# Manually verify environment
micromamba run -p envs/conda_cache/drugsol-train \
    python -c "import torch, xgboost, lightgbm; print('OK')"
```

---

## Citation

If you use DrugSol in your research, please cite:

```bibtex
@software{drugsol2024,
  author = {Olivares Rodriguez, Aitor},
  title = {DrugSol: Machine Learning Pipeline for Aqueous Solubility Prediction},
  year = {2024},
  url = {https://github.com/yourusername/drugsol}
}
```

### Related Publications

- **BigSolDB**: [Zenodo Record 15094979](https://zenodo.org/records/15094979)
- **Chemprop**: Yang et al. (2019) "Analyzing Learned Molecular Representations for Property Prediction" *J. Chem. Inf. Model.*
- **QED**: Bickerton et al. (2012) "Quantifying the chemical beauty of drugs" *Nature Chemistry*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit changes (`git commit -am 'Add new model'`)
4. Push to branch (`git push origin feature/new-model`)
5. Open a Pull Request

---

## Acknowledgments

- **Universitat Rovira i Virgili** - Academic supervision
- **BigSolDB** - Primary solubility dataset
- **ChEMBL** - Secondary data source
- **Chemprop** - Graph neural network implementation
- **Nextflow** - Workflow management
