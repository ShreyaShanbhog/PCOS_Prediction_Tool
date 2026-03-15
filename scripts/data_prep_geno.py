import pandas as pd
import glob
import os

# -----------------------------
# 1. Load metadata
# -----------------------------

metadata = pd.read_excel("data/metadata.xlsx")

# Keep relevant columns
metadata = metadata[["ID", "Disease"]]

# Standardize labels
metadata["Disease"] = metadata["Disease"].str.lower()

print("Metadata loaded")
print(metadata.head())

# -----------------------------
# 2. Locate gene expression files
# -----------------------------

expression_files = glob.glob("data/*.csv")

print(f"Found {len(expression_files)} expression files")

# -----------------------------
# 3. Process each file
# -----------------------------

all_samples = []

for file in expression_files:

    print(f"\nProcessing: {file}")

    df = pd.read_csv(file)
    df = df.set_index("GB_ACC")

    # Remove non-expression columns
    df = df.drop(columns=["ID"], errors="ignore")

    # Transpose
    df_t = df.T
    df_t = df_t.loc[:, ~df_t.columns.duplicated()]

    # Convert index to sample ID column
    df_t.index.name = "ID"
    df_t.reset_index(inplace=True)

    # Merge with metadata
    df_t = df_t.merge(metadata, on="ID", how="left")

    # Rename label column
    df_t = df_t.rename(columns={"Disease": "Label"})

    # Remove samples without labels
    df_t = df_t.dropna(subset=["Label"])

    print(f"Samples retained: {df_t.shape[0]}")

    all_samples.append(df_t)

# -----------------------------
# 4. Combine all datasets
# -----------------------------

final_df = pd.concat(all_samples, ignore_index=True)

print("\nFinal dataset shape:")
print(final_df.shape)

# -----------------------------
# 5. Check class distribution
# -----------------------------

print("\nClass distribution:")
print(final_df["Label"].value_counts())

# -----------------------------
# 6. Save processed dataset
# -----------------------------

final_df.to_csv("processed_gene_expression.csv", index=False)

print("\nSaved file: processed_gene_expression.csv")
