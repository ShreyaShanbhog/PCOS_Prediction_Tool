import os
import sys

# Allow Python to find ../modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from modules.preprocessing import extract_expression_matrix, save_cleaned_data


def main():
    input_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "GSE10946_cumulus_cell.txt"
    )

    output_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "GSE10946_cumulus_cell_exp-mat.csv"
    )

    print(f"Reading raw GEO file:\n  {input_file}")
    df = extract_expression_matrix(input_file)

    if df.empty:
        print("❌ No expression matrix extracted.")
        sys.exit(1)

    print(f"✅ Extracted matrix with shape {df.shape}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_cleaned_data(df, output_file)

    print(f"💾 Saved cleaned matrix to:\n  {output_file}")


if __name__ == "__main__":
    main()

