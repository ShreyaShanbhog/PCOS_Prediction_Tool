# PCOS Subtype & Trajectory Prediction Tool

This project analyzes **gene expression data** and **medical images** to classify **PCOS subtypes** and predict **disease**. It combines computational biology and machine learning.

## Features
- Integrates gene expression and imaging data.
- Identifies differentially expressed genes.
- Generates UMAP/PCA visualizations.
- Predicts PCOS subtypes: insulin-resistant, hormonal imbalance, lifestyle, etc.
- Provides exploratory plots and enrichment analysis.

## Structure
- `data/` → input datasets (gene expression + images)
- `scripts/` → runnable scripts
- `modules/` → functions for analysis
- `results/` → plots, predictions, outputs

## Quick Start
```bash
pip install -r requirements.txt
python scripts/run_analysis.py
