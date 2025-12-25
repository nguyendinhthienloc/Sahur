import pandas as pd
from pathlib import Path
import re
import shutil
import sys

# --- CONFIGURATION ---
# Input directory containing the topic CSVs
INPUT_DIR = Path("data/cleaned_by_topic")

# Output directory for the fixed files
OUTPUT_DIR = Path("data/cleaned_patched")

# List of all text columns to check and fix
# (Includes the human column and all AI model columns identified in your files)
TARGET_COLUMNS = [
    'Human_story', 
    'llama-8B', 
    'mistral-7B', 
    'gemma-2-9b', 
    'GPT_4-o', 
    'qwen-2-72B', 
    'yi-large',
    'prompt'  # Good idea to clean the prompt too if it has similar issues
]

def fix_mashups(text):
    """
    Applies regex patterns to split mashed-up words.
    """
    if pd.isna(text):
        return text
    
    # Ensure text is a string
    text = str(text)
    
    # FIX 1: Split CamelCase (e.g., "TimesUpdated" -> "Times Updated")
    # Logic: Lowercase letter followed immediately by an Uppercase letter
    # We ignore the start of the string to avoid affecting the very first word if weirdly capitalized
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # FIX 2: Split Period Mashups (e.g., "sentence.The" -> "sentence. The")
    # Logic: Lowercase letter, literal dot, immediate Uppercase letter
    text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', text)
    
    # FIX 3: Split Comma Mashups (e.g., "city,State" -> "city, State")
    # Logic: Lowercase letter, literal comma, immediate Uppercase letter
    text = re.sub(r'([a-z]),([A-Z])', r'\1, \2', text)
    
    return text

def main():
    # 1. Validation
    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory '{INPUT_DIR}' does not exist.")
        print("Please run this script from the project root.")
        sys.exit(1)

    # 2. Setup Output Directory
    if OUTPUT_DIR.exists():
        print(f"Cleaning existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    print(f"Output directory created: {OUTPUT_DIR}\n")

    # 3. Process Files
    csv_files = list(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        print(f"WARNING: No CSV files found in {INPUT_DIR}")
        sys.exit(0)

    print(f"Found {len(csv_files)} files to process.")
    print("-" * 60)

    for i, file_path in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] Processing {file_path.name}...", end=" ", flush=True)
        
        try:
            # Read the CSV
            df = pd.read_csv(file_path)
            
            # Identify which columns in this specific file need fixing
            cols_to_fix = [col for col in TARGET_COLUMNS if col in df.columns]
            
            if not cols_to_fix:
                print(f"\n    Skipping (No target columns found). Columns: {list(df.columns)}")
                continue
            
            # Apply the fix to each column
            changes_count = 0
            for col in cols_to_fix:
                # We apply the fix and just overwrite the column
                # (Optional: You could count changes here if you really wanted to, but it slows it down)
                df[col] = df[col].apply(fix_mashups)
            
            # Save the fixed dataframe to the new folder
            output_path = OUTPUT_DIR / file_path.name
            df.to_csv(output_path, index=False)
            print("Done.")

        except Exception as e:
            print(f"\n    ERROR processing file: {e}")

    print("-" * 60)
    print("Processing complete!")
    print(f"Cleaned files are located in: {OUTPUT_DIR}")
    print("Next Step: Update your analysis scripts to read from this new directory.")

if __name__ == "__main__":
    main()