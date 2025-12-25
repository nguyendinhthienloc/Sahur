import pandas as pd
import glob
import os
import re

# ================= CONFIGURATION =================
INPUT_PATTERN = 'best25_*.csv'
OUTPUT_FOLDER = './repaired_best25'
# =================================================

def repair_text(text):
    if not isinstance(text, str):
        return text
    
    # Work with a version we can check for patterns
    clean_text = text.strip()
    
    # --- FIX 1: Video/Transcript Artifacts ---
    # Example: "new video loaded: Title transcript Title Actual Story..."
    # Logic: If we find "transcript", distinct split. If not, regex replace the prefix.
    
    if "new video loaded" in clean_text.lower():
        # Try to split by "transcript" (case insensitive)
        # This usually separates the metadata garbage from the real text
        parts = re.split(r'transcript', clean_text, flags=re.IGNORECASE, maxsplit=1)
        
        if len(parts) > 1:
            # We found 'transcript', take the part AFTER it
            clean_text = parts[1].strip()
        else:
            # No 'transcript' found, just strip the "new video loaded:" prefix
            clean_text = re.sub(r'^new video loaded:?\s*', '', clean_text, flags=re.IGNORECASE).strip()

    # --- FIX 2: Capitalization (The "Mashup" look) ---
    # If the text starts with a lowercase letter, capitalize it.
    if len(clean_text) > 0 and clean_text[0].islower():
        clean_text = clean_text[0].upper() + clean_text[1:]

    return clean_text

def process_repairs():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files = glob.glob(INPUT_PATTERN)
    print(f"Found {len(files)} files to repair.\n")

    for filepath in files:
        filename = os.path.basename(filepath)
        
        try:
            df = pd.read_csv(filepath)
            
            # Identify the text column
            col = 'Human_story' if 'Human_story' in df.columns else None
            if not col:
                # Fallback search
                for c in df.columns:
                    if 'story' in c.lower() or 'text' in c.lower():
                        col = c
                        break
            
            if col:
                # Apply the repair function to every row in that column
                # We are NOT dropping rows, we are modifying them.
                df[col] = df[col].apply(repair_text)
                
                # Save to new folder
                output_path = os.path.join(OUTPUT_FOLDER, filename)
                df.to_csv(output_path, index=False)
                
                print(f"[REPAIRED] {filename}: Processed {len(df)} rows.")
            else:
                print(f"[SKIP] {filename}: Could not find text column.")

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

if __name__ == "__main__":
    process_repairs()