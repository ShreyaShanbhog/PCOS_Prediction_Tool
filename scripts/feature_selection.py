import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# -----------------------------
# 1. Load datasets
# -----------------------------

print("Loading gene expression data...")

data = pd.read_csv("data/processed_gene_expression.csv")

print("Loading metadata...")

metadata = pd.read_excel("data/metadata.xlsx")

metadata = metadata[["ID", "Disease"]]
metadata["Disease"] = metadata["Disease"].str.lower()

# -----------------------------
# 2. Merge labels with expression data
# -----------------------------

print("Merging labels...")

data = data.merge(metadata, on="ID", how="left")

# Remove samples without labels
data = data.dropna(subset=["Disease"])

print("Dataset after merge:", data.shape)

print("\nClass distribution:")
print(data["Disease"].value_counts())

# -----------------------------
# 3. Separate features and labels
# -----------------------------

X = data.drop(columns=["ID", "Disease"])
y = data["Disease"]

print("\nFeature matrix shape:", X.shape)

# -----------------------------
# 4. Apply feature selection
# -----------------------------

print("\nSelecting top 300 genes using ANOVA...")

selector = SelectKBest(score_func=f_classif, k=300)

X_new = selector.fit_transform(X, y)

# -----------------------------
# 5. Get selected gene names
# -----------------------------

selected_genes = X.columns[selector.get_support()]

print("\nNumber of selected genes:", len(selected_genes))

print("\nExample selected genes:")
print(selected_genes[:10])

# -----------------------------
# 6. Create reduced dataset
# -----------------------------

X_selected = pd.DataFrame(X_new, columns=selected_genes)

# Add ID and label columns back
X_selected["ID"] = data["ID"].values
X_selected["Disease"] = y.values

# Reorder columns
cols = ["ID", "Disease"] + list(selected_genes)
X_selected = X_selected[cols]

print("\nFinal dataset shape:", X_selected.shape)

# -----------------------------
# 7. Save dataset
# -----------------------------

output_file = "gene_features_selected.csv"

X_selected.to_csv(output_file, index=False)

print("\nSaved reduced dataset to:", output_file)
