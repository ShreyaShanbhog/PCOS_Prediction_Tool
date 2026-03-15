#!/usr/bin/env python

import numpy as np
import pandas as pd


def get_top_genes(model, gene_names, top_k=20):
    """
    Identify the top important genes from a trained model.

    Parameters
    ----------
    model : trained ML model
        A fitted classifier with attribute `feature_importances_`
        (e.g., RandomForest, XGBoost).

    gene_names : list or array-like
        Names of the gene features used during model training.

    top_k : int, default=20
        Number of top genes to return.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the top genes and their importance scores.
        Columns: ['gene', 'importance']
    """
