# Dataset Overview

## Gene Expression Data
- RNA-seq data from healthy and PCOS patients.
- ~30 samples (combined from 2 datasets).
- Used for differential expression, clustering, and subtype prediction.

## Image Data
- Ovarian/ultrasound images.
- Used for image-based feature extraction and integration with gene data.

## Data Processing
- Gene expression: QC, normalization, batch correction.
- Images: preprocessing, resizing, augmentation.
- Both datasets mapped to the same sample IDs for integration.

## Example dataset for using the tool

This example dataset consists of a gene expression matrix with 91 samples and 302 genes. Each row represents a sample, and each column corresponds to a gene expression value. The dataset includes both PCOS and control samples, with labels indicating disease status.

Basic statistics:
- Number of samples: 91
- Number of genes: 302
- Data type: normalized gene expression values

This dataset is a good example for running the tool because it is small enough for fast computation while still capturing meaningful biological variation between PCOS and control groups. It allows testing of the full pipeline, including model training and feature importance extraction, without requiring large computational resources.

The dataset is stored in the `data/` folder of the repository.
