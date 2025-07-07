import pandas as pd
import re
from typing import List

def load_complaints(file_path: str) -> pd.DataFrame:
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, dtype={'ZIP code': str})
    print(f"Loaded {len(df)} rows.")
    return df

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove dates and other common boilerplate
    text = re.sub(r'xx/xx/\d{4}', '', text)
    text = re.sub(r'xx/xx/xxxx', '', text)
    text = re.sub(r'xxxx', '', text)
    # Remove special characters but keep sentences intact
    text = re.sub(r'[^a-z0-9\s\.\?,!]', '', text)
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_data(df: pd.DataFrame, products: List[str]) -> pd.DataFrame:
    print("Starting preprocessing...")
    
    # 1. Rename columns for easier access
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # 2. Filter for specified products
    print(f"Filtering for products: {', '.join(products)}")
    initial_rows = len(df)
    df_filtered = df[df['product'].isin(products)].copy()
    print(f"  - Removed {initial_rows - len(df_filtered)} rows with other products.")
    
    # 3. Filter out rows with missing narratives
    df_filtered = df_filtered.rename(columns={'consumer_complaint_narrative': 'narrative'})
    initial_rows = len(df_filtered)
    df_filtered.dropna(subset=['narrative'], inplace=True)
    df_filtered = df_filtered[df_filtered['narrative'].str.strip() != '']
    print(f"  - Removed {initial_rows - len(df_filtered)} rows with empty narratives.")
    
    # 4. Clean the text narratives
    print("Cleaning text narratives...")
    df_filtered['narrative_cleaned'] = df_filtered['narrative'].apply(clean_text)
    
    # 5. Select and reorder final columns
    final_cols = ['complaint_id', 'product', 'issue', 'company', 'state', 'narrative', 'narrative_cleaned']
    df_final = df_filtered[[col for col in final_cols if col in df_filtered.columns]].copy()
    
    print(f"Preprocessing complete. Final dataset has {len(df_final)} rows.")
    return df_final