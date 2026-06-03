# PCOS and Subtype Prediction Tool

This project analyzes **gene expression data** and **medical images** to classify **PCOS subtypes** and predict disease. It combines computational biology and machine learning, with a Streamlit-based UI deployed on an HPC cluster.

## Features

- Integrates gene expression and imaging data
- Identifies differentially expressed genes
- Generates UMAP/PCA visualizations
- Predicts PCOS subtypes: insulin-resistant, hormonal imbalance, lifestyle, etc.
- Provides exploratory plots and enrichment analysis

## Repository Structure

```
PCOS_Prediction_Tool/
├── data/            # Input datasets (gene expression + images)
├── scripts/         # Runnable analysis scripts
├── requirements.txt
├── dataset.md       # Dataset details
├── dds.md           # Differential expression design
└── wbs.md           # Work breakdown structure
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python scripts/run_analysis.py

# 3. Launch the Streamlit UI
streamlit run app.py
```

## Dataset

The example dataset is available in the `data/` folder. See [`dataset.md`](dataset.md) for full details.

## License

_Add license here._
