import pandas as pd
import glob
import re
import os

# --- CONFIGURATION ---
INPUT_PATTERN = 'best25_*.csv'  # Process ALL best25 files in the folder

# Whitelist: Valid words > 20 chars to PRESERVE
WHITELIST = {
    'Thiruvenkatanathapuram', 'mischaracterizations', 'counterrevolutionaries',
    'electroencephalogram', 'internationalization', 'compartmentalization',
    'uncharacteristically', 'indistinguishability', 'institutionalization',
    'gastroenterologist', 'bioluminescent', 'telecommunications',
    'pathophysiological', 'immunohistochemical'
}

def polish_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Identify all long alphabetic strings (>19 chars)
    # This catches things like "nytopinionatorontwitter" (23 chars)
    long_tokens = re.findall(r'\b[a-zA-Z]{20,}\b', text)
    
    for token in long_tokens:
        # If it's not a known valid word, remove it
        if token not in WHITELIST:
            # Replace with space (safer than empty string to avoid merging neighbors)
            text = text.replace(token, " ")
            
    # 2. Fix CamelCase Mashups (e.g., "Furstenbergannounced" -> "Furstenberg announced")
    # Logic: Lowercase letter followed immediately by Uppercase letter
    # We replace it with "Lower Upper"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # 3. Fix Period Mashups (e.g., "end.Start" -> "end. Start")
    text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', text)

    # 4. Cleanup extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_polish():
    files = glob.glob(INPUT_PATTERN)
    print(f"Found {len(files)} files to polish...")
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'Human_story' not in df.columns:
                continue
                
            # Apply the cleaning
            df['Human_story'] = df['Human_story'].apply(polish_text)
            
            # Save back to the same file
            df.to_csv(f, index=False)
            print(f"  ✓ Cleaned & Saved: {f}")
            
        except Exception as e:
            print(f"  ✗ Error in {f}: {e}")

if __name__ == "__main__":
    run_polish()