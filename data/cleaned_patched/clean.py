import pandas as pd
import glob
import re
import os

# --- CONFIGURATION ---
INPUT_FOLDER = '.'          # Folder containing original CSVs
OUTPUT_FOLDER = 'data'      # Target folder
OUTPUT_PREFIX = 'best25_'   
MIN_WORD_COUNT = 200        

# Whitelist of valid long words to preserve
VALID_LONG_WORDS = {
    'autobiographical', 'characterization', 'counterintuitive', 'counterproductive', 
    'demilitarization', 'disproportionate', 'disproportionately', 'electrocardiogram', 
    'environmentalists', 'gastroenterologist', 'gastroenterology', 'gastrointestinal', 
    'hospitalizations', 'hydroxychloroquine', 'industrialization', 'institutionalize', 
    'insurrectionists', 'intercontinental', 'interrelationships', 'irresponsibility', 
    'multiculturalism', 'multidimensional', 'photojournalists', 'responsibilities', 
    'responsibility', 'straightforwardly', 'transmissibility', 'unconventionality', 
    'unpredictability', 'telecommunications', 'underrepresentation', 'misrepresentation', 
    'epidemiologist', 'anesthesiologist', 'acknowledgments', 'characteristically',
    'uncharacteristically', 'incomprehensible', 'interdisciplinary', 'bioluminescent',
    'enthusiastically', 'simultaneously', 'extraordinary', 'approximately', 'congratulations'
}

def clean_text_refined(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Remove boilerplate
    boilerplate = [
        "Slideshow controls", "Site Search Navigation", "Site Navigation", 
        "Site Mobile Navigation", "Supported by", "skip to content", 
        "View on Instagram", "Read the article", "Advertisement"
    ]
    for phrase in boilerplate:
        text = text.replace(phrase, "")
    
    # 2. Fix mashed spacing using Regex
    text = re.sub(r'([.,;:])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r"(['’]s)([a-zA-Z])", r"\1 \2", text)
    
    # 3. Basic cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_bad_mashups(text):
    """Returns a list of bad mashup words found in the text."""
    words = text.split()
    bad_ones = []
    for w in words:
        # Clean punctuation to check pure word length
        w_clean = re.sub(r'^[^\w]+|[^\w]+$', '', w) 
        # Check for lowercase words > 15 chars that are NOT in whitelist
        if len(w_clean) > 15 and w_clean.isalpha() and w_clean.islower():
            if w_clean not in VALID_LONG_WORDS:
                bad_ones.append(w)
    return bad_ones

def process_folder():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Find input files
    search_path = os.path.join(INPUT_FOLDER, "*.csv")
    files = glob.glob(search_path)
    # Filter to avoid processing output files
    files = [f for f in files if "cleaned_" not in f and OUTPUT_PREFIX not in f and "data" not in f]
    
    print(f"Processing {len(files)} files...")

    for f in files:
        filename = os.path.basename(f)
        try:
            df = pd.read_csv(f)
            if 'Human_story' not in df.columns: continue
            
            # Step 1: Initial Clean
            df['Human_story_clean'] = df['Human_story'].apply(clean_text_refined)
            df['word_count'] = df['Human_story_clean'].apply(lambda x: len(str(x).split()))
            
            # Step 2: Filter Candidates > 200 words
            df_candidates = df[df['word_count'] >= MIN_WORD_COUNT].copy()
            
            # Step 3: Prioritize Clean Rows
            clean_rows = []
            repaired_rows = []
            
            for idx, row in df_candidates.iterrows():
                text = row['Human_story_clean']
                bad_words = get_bad_mashups(text)
                
                if not bad_words:
                    clean_rows.append(row)
                else:
                    # Repair row by removing bad words
                    for bw in bad_words:
                        text = text.replace(bw, "")
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    # Only keep if it still meets length requirement
                    if len(text.split()) >= MIN_WORD_COUNT:
                        row['Human_story_clean'] = text
                        repaired_rows.append(row)
            
            # Step 4: Select Top 25
            df_clean = pd.DataFrame(clean_rows)
            df_repaired = pd.DataFrame(repaired_rows)
            
            final_selection = pd.DataFrame()
            
            # First pick from perfect rows
            if not df_clean.empty:
                df_clean = df_clean.sort_values(by='word_count', ascending=False)
                final_selection = pd.concat([final_selection, df_clean])
            
            # If needed, fill with repaired rows
            if len(final_selection) < 25 and not df_repaired.empty:
                df_repaired['word_count'] = df_repaired['Human_story_clean'].apply(lambda x: len(str(x).split()))
                df_repaired = df_repaired.sort_values(by='word_count', ascending=False)
                
                needed = 25 - len(final_selection)
                final_selection = pd.concat([final_selection, df_repaired.head(needed)])
            
            # Truncate to exactly 25
            final_selection = final_selection.head(25)
            
            # Save
            if not final_selection.empty:
                output_df = final_selection.copy()
                output_df['Human_story'] = output_df['Human_story_clean']
                
                # Keep original columns
                cols = [c for c in df.columns if c not in ['Human_story_clean', 'word_count']]
                output_df = output_df[cols]
                
                # Naming: best25_filename.csv
                clean_name = filename.replace("cleaned_", "")
                out_path = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_PREFIX}{clean_name}")
                output_df.to_csv(out_path, index=False)
                print(f"Saved {len(output_df)} rows to {out_path}")
            
        except Exception as e:
            print(f"Error {filename}: {e}")

if __name__ == "__main__":
    process_folder()