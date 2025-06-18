import pandas as pd
import sqlite3
import os

# --- Configuration ---
CURRENT_DIR = os.path.dirname(__file__)
SOURCE_RESTAURANT_PATH = os.path.join(CURRENT_DIR, '..', 'source_data', 'restaurant_categorized_withcompscore.xlsx')
SOURCE_BOUTIQUE_PATH = os.path.join(CURRENT_DIR, '..', 'source_data', 'boutique_rankings.csv')
DATABASE_DIR = os.path.join(CURRENT_DIR, '..', 'database')
DATABASE_PATH = os.path.join(DATABASE_DIR, 'recommender.db')

# --- Helper function for normalization ---
def _safe_normalise(series: pd.Series) -> pd.Series:
    """Safely normalizes a pandas Series to a 0-1 scale, handling non-numeric data."""
    numeric_series = pd.to_numeric(series, errors='coerce').fillna(0)
    lo, hi = numeric_series.min(), numeric_series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (numeric_series - lo) / (hi - lo)

# --- CORRECTED Restaurant Scoring Algorithm ---
def calculate_restaurant_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the composite score for restaurants using the EXACT column names
    from your provided index.
    """
    print("Calculating scores for restaurants using precise formula and correct column names...")

    # Using the exact column names you provided.
    pop = df['Popularity_Score'].fillna(0)
    amb = df['Ambiance_Score'].fillna(0)
    serv = df['Service_Score'].fillna(0)
    uniq = df['Uniqueness_Score'].fillna(0)
    hyg_flag = df['Hygiene_Certification'].fillna(0)
    
    # CRITICAL FIX: Using the correct 'NRI_Friendly_Score' column name.
    nri_flag = df['NRI_Friendly_Score'].fillna(0)

    # Apply the formula directly
    df['Raw_Score'] = (
        0.389 * pop +
        0.278 * amb +
        0.259 * serv +
        0.082 * uniq +
        0.190 * hyg_flag +  # This is 0.019 * 10
        0.140 * nri_flag +  # This is 0.014 * 10
        0.012
    )

    # Calculate the final composite score
    df['Composite_New'] = df['Raw_Score'] * 10
    
    return df

# --- CORRECTED Boutique Scoring Algorithm ---
def calculate_boutique_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the composite score for boutiques using the EXACT column names
    from your provided index.
    """
    print("Calculating scores for boutiques using precise formula and correct column names...")
    
    weights = {'rating': 0.20, 'reviews': 0.10, 'sentiment': 0.10, 'followers': 0.15, 'mentions': 0.10, 'website': 0.05, 'style': 0.30}
    
    # Use the exact column names you provided for normalization
    df['Rating_norm_calc'] = _safe_normalise(df['Google Rating'])
    df['Reviews_norm_calc'] = _safe_normalise(df['Review Count'])
    df['Sentiment_norm_calc'] = _safe_normalise(df['Avg Sentiment Score'])
    df['Followers_norm_calc'] = _safe_normalise(df['Instagram Followers'])
    df['Mentions_norm_calc'] = _safe_normalise(df['Blog Mentions'])
    df['Website_norm_calc'] = df['Has Website'].apply(lambda x: 1 if isinstance(x, str) and x.lower() == 'yes' else 0).fillna(0)
    df['Uniqueness_norm_calc'] = _safe_normalise(df['Style Uniqueness Score'])
    
    # Calculate the composite score using the newly calculated normalized values
    df['Composite Score_calc'] = (
        df['Rating_norm_calc'] * weights['rating'] +
        df['Reviews_norm_calc'] * weights['reviews'] +
        df['Sentiment_norm_calc'] * weights['sentiment'] +
        df['Followers_norm_calc'] * weights['followers'] +
        df['Mentions_norm_calc'] * weights['mentions'] +
        df['Website_norm_calc'] * weights['website'] +
        df['Uniqueness_norm_calc'] * weights['style']
    )
    return df

def run_the_pipeline():
    """Main pipeline function."""
    print("--- Starting Data Pipeline ---")

    # 1. Load the source data
    restaurants_df = pd.read_excel(SOURCE_RESTAURANT_PATH)
    boutiques_df = pd.read_csv(SOURCE_BOUTIQUE_PATH)
    
    # 2. Calculate the scores using the corrected functions
    restaurants_with_scores = calculate_restaurant_scores(restaurants_df)
    boutiques_with_scores = calculate_boutique_scores(boutiques_df)

    # 3. Write the final, processed data to the database
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        print("Writing final 'restaurants' table to database...")
        restaurants_with_scores.to_sql('restaurants', conn, if_exists='replace', index=False)
        
        print("Writing final 'boutiques' table to database...")
        boutiques_with_scores.to_sql('boutiques', conn, if_exists='replace', index=False)
    finally:
        conn.close()

    print("\n--- Data Pipeline Finished Successfully! ---")
    print("Database `recommender.db` has been correctly created with your precise formulas.")

if __name__ == "__main__":
    run_the_pipeline()