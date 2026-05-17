import pandas as pd
import glob
import os

# --- CONFIGURATION ---
# Path to the folder containing your 9 CSV files
folder_path = "C:/Users/jakad/OneDrive/Desktop/ds2023/btp_2025/csv_files"

# Output file name
output_file = "C:/Users/jakad/OneDrive/Desktop/ds2023/btp_2025/full_reviews.csv"

# --- MERGE SCRIPT ---

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Check if any CSV files found
if not csv_files:
    print("No CSV files found in the specified folder.")
else:
    print(f"Found {len(csv_files)} CSV files. Merging...")

    # Read and concatenate all CSVs
    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    # Optionally remove duplicates
    merged_df.drop_duplicates(inplace=True)

    # Save to a new CSV
    merged_df.to_csv(output_file, index=False)

    print(f"✅ Merged file saved as '{output_file}' with {len(merged_df)} rows.")
