import pandas as pd
import io

def extract_expression_matrix(file_path):
    """
    Extracts the data between !series_matrix_table_begin and !series_matrix_table_end.
    This function filters out GEO metadata and returns a cleaned DataFrame.
    """
    extracted_lines = []
    capture = False

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Logic branch: check for the end tag first to stop capturing
                if "!series_matrix_table_end" in line:
                    capture = False
                    break
                
                # Logic branch: if capture is active, save the line
                if capture:
                    extracted_lines.append(line)
                
                # Logic branch: check for the start tag
                if "!series_matrix_table_begin" in line:
                    capture = True

        if not extracted_lines:
            return pd.DataFrame()

        # Convert the list of strings into a DataFrame
        data_str = "".join(extracted_lines)
        df = pd.read_csv(io.StringIO(data_str), sep='\t', quotechar='"')
        return df

    except FileNotFoundError:
        print("Error: File not found.")
        return pd.DataFrame()

def save_cleaned_data(df, output_path):
    """
    Saves the cleaned expression matrix to a CSV file.
    Returns True if successful, False otherwise.
    """
    # Branch: Check if there is actually data to save
    if df is not None and not df.empty:
        df.to_csv(output_path, index=False)
        return True
    
    # Branch: Edge case where DF is empty
    return False
    
