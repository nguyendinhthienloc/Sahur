import pandas as pd
import glob
import os
import re

# ================= CONFIGURATION =================
# Point this to the folder containing your cleaned/repaired files
INPUT_FOLDER = r'./repaired_best25'
# =================================================

def check_cleanliness():
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    print(f"Inspecting {len(files)} files in {INPUT_FOLDER}...\n")
    
    total_issues = 0
    
    for filepath in files:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        
        # Identify text column
        col = 'Human_story' if 'Human_story' in df.columns else None
        if not col:
            for c in df.columns:
                if 'story' in c.lower() or 'text' in c.lower():
                    col = c
                    break
        
        if col:
            file_issues = 0
            for idx, row in df.iterrows():
                text = str(row[col]).strip()
                
                # CHECK 1: Video/Transcript Artifacts
                if "new video loaded" in text.lower():
                    print(f"  [FAIL] {filename} Row {idx}: Contains 'new video loaded'")
                    file_issues += 1
                elif text.lower().startswith("transcript"):
                    print(f"  [FAIL] {filename} Row {idx}: Starts with 'transcript'")
                    file_issues += 1
                
                # CHECK 2: Lowercase Start (Mashup)
                elif len(text) > 0 and text[0].islower():
                     print(f"  [FAIL] {filename} Row {idx}: Starts with lowercase ('{text[:10]}...')")
                     file_issues += 1
            
            if file_issues == 0:
                print(f"[PASS] {filename}")
            else:
                total_issues += file_issues
        else:
            print(f"[SKIP] {filename}: Column not found")

    print("-" * 30)
    if total_issues == 0:
        print("\n✅ SUCCESS: Data is clean. Ready for metrics extraction.")
    else:
        print(f"\n❌ FAILED: Found {total_issues} issues remaining.")

if __name__ == "__main__":
    check_cleanliness()