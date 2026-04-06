# Dataset Overview

## Gene Expression Data
- RNA-seq data from healthy and PCOS patients.
- ~90 samples (combined from datasets).
- Used for differential expression, clustering, and subtype prediction.

## Image Data
- Ovarian/ultrasound images.
- Used for image-based feature extraction and integration with gene data.

## Data Processing
- Gene expression: QC, normalization, batch correction.
- Images: preprocessing, resizing, augmentation.
- Gene expression and image data are aligned by sample IDs for multimodal analysis.

## Example dataset for using the tool

This example dataset consists of a gene expression matrix with 91 samples and 302 genes. Each row represents a sample, and each column corresponds to a gene expression value. The dataset includes both PCOS and control samples, with labels indicating disease status.

Basic statistics:
- Number of samples: 91
- Number of genes: 302
- Data type: normalized gene expression values

This dataset is a good example for running the tool because it is small enough for fast computation while still capturing meaningful biological variation between PCOS and control groups. It allows testing of the full pipeline, including model training and feature importance extraction, without requiring large computational resources.

The dataset is stored in the `data/` folder of the repository.

## Real dataset for answering a biological question using the tool

We use a publicly available gene expression dataset related to Polycystic Ovary Syndrome (PCOS) from the Gene Expression Omnibus (GEO):

Dataset: GSE34526  
URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE34526  

This dataset contains gene expression profiles from women diagnosed with PCOS and healthy controls.

**Dataset characteristics:**
- Number of samples: ~90  
- Number of genes: ~30,000 (reduced to ~300 after preprocessing)  
- Structure: rows = samples, columns = genes  
- Labels: PCOS (1) vs Control (0)  

**Why this dataset is suitable:**
This dataset is directly relevant to PCOS and contains labeled case-control data, making it appropriate for supervised learning. Its high dimensionality makes it suitable for testing the tool’s ability to perform feature selection and interpretability analysis.

**Biological question:**
Which genes are most predictive of PCOS, and can gene expression data be used to accurately classify patients as PCOS or non-PCOS?

**Expected results:**
- A trained classification model (e.g., Random Forest)  
- Model performance metrics (accuracy, ROC-AUC)  
- A ranked list of top predictive genes (e.g., top 20 genes)  

**Expected answer:**
We expect to identify a subset of genes that strongly differentiate PCOS from control samples. These genes may correspond to known biological pathways or serve as potential biomarkers for PCOS.
