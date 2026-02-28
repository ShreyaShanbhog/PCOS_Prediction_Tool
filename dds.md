# Design Document Specification (DDS)

## Goal
Predict PCOS  and its subtypes using gene expression + image data.

## Modules (Functions)
- `modules/preprocessing.py → cleans the raw files

## Scripts
- `scripts/preprocessing_geo.py` → runs the function

## Inputs
- Gene expression matrix
- Preprocessed images
- Sample metadata

## Outputs
- Cluster labels & predicted subtypes
- Dimensionality reduction plots
- Enrichment analysis & feature summary
