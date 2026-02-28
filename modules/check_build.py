import os
from modules.preprocessing import extract_expression_matrix, save_cleaned_data

def run_pipeline():
    # Define paths
    input_dir = "data"
    output_dir = "data" # You could also create a 'data/cleaned' folder
    
    # List of your specific project files
    geo_files = [
        "GSE1615-GPL96_series_matrix.txt",
        "GSE1615-GPL97_series_matrix.txt",
        "GSE199225_series_matrix.txt",
        "GSE34526_series_matrix.txt"
    ]

    print(f"--- Starting Preprocessing Pipeline ---")

    for filename in geo_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".txt", "_cleaned.csv"))

        if os.path.exists(input_path):
            print(f"Processing: {filename}...")
            
            # 1. Extract the matrix (The "Meaningful Implementation")
            df = extract_expression_matrix(input_path)
            
            # 2. Save the result (The "Suitable Format")
            if not df.empty:
                success = save_cleaned_data(df, output_path)
                if success:
                    print(f"Successfully saved to: {output_path}")
            else:
                print(f"Warning: No data found in {filename}")
        else:
            print(f"Error: {filename} not found in {input_dir}")

if __name__ == "__main__":
    run_pipeline()
