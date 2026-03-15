import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------

print("Loading selected gene dataset...")

data = pd.read_csv("data/gene_features_selected.csv")

print("Dataset shape:", data.shape)

# -----------------------------
# 2. Separate features and labels
# -----------------------------

X = data.drop(columns=["ID", "Disease"])
y = data["Disease"]

# -----------------------------
# 3. Train/test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -----------------------------
# 4. Train Random Forest model
# -----------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Cross Validation
# ----------------------------

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Mean CV accuracy:", scores.mean())

# -----------------------------
# 5. Evaluate model
# -----------------------------

y_pred = model.predict(X_test)

print("\nModel Performance")

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 6. Save model
# -----------------------------

joblib.dump(model, "pcos_genomic_model.pkl")

print("\nModel saved as: pcos_genomic_model.pkl")

# -----------------------------
# 7. Identify important genes
# -----------------------------

importances = model.feature_importances_

feature_importance = pd.DataFrame({
    "Gene": X.columns,
    "Importance": importances
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Genes:")
print(feature_importance.head(10))

feature_importance.to_csv("gene_importance.csv", index=False)

print("\nSaved gene importance to: gene_importance.csv")

# -----------------------------
# 8. visualization
# -----------------------------
import matplotlib.pyplot as plt

top = feature_importance.head(20)

plt.figure(figsize=(8,6))
plt.barh(top["Gene"], top["Importance"])
plt.gca().invert_yaxis()
plt.title("Top 20 PCOS Predictive Genes")
plt.xlabel("Importance")

plt.tight_layout()
plt.savefig("gene_importance_plot.png")
