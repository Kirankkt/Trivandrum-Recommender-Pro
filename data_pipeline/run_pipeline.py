import pandas as pd
import sqlite3
import os
import json
import time
import google.generativeai as genai

# --- Configuration ---
CURRENT_DIR = os.path.dirname(__file__)
# The pipeline now reads from this simple "to-do list" file.
BASE_VENUES_PATH = os.path.join(CURRENT_DIR, 'base_venues.csv')
DATABASE_DIR = os.path.join(CURRENT_DIR, '..', 'database')
DATABASE_PATH = os.path.join(DATABASE_DIR, 'recommender.db')

# --- Gemini API Configuration ---
# This uses the secret key you set in GitHub and Hugging Face.
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception as e:
    print(f"CRITICAL: Gemini API key not configured. Please set the GEMINI_API_KEY secret. Error: {e}")

# --- Helper and Scoring Functions (Unchanged) ---
def _safe_normalise(series: pd.Series) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors='coerce').fillna(0)
    lo, hi = numeric_series.min(), numeric_series.max()
    if hi == lo: return pd.Series(0.5, index=series.index)
    return (numeric_series - lo) / (hi - lo)

def calculate_restaurant_scores(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating scores for restaurants...")
    pop = df['popularity_score'].fillna(0); amb = df['ambiance_score'].fillna(0); serv = df['service_score'].fillna(0); uniq = df['uniqueness_score'].fillna(0); hyg_flag = df['hygiene_certification'].fillna(0); nri_flag = df['nri_friendly_score'].fillna(0)
    df['raw_score'] = (0.389*pop + 0.278*amb + 0.259*serv + 0.082*uniq + 0.190*hyg_flag + 0.140*nri_flag + 0.012)
    df['composite_new'] = df['raw_score'] * 10
    return df

def calculate_boutique_scores(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating scores for boutiques...")
    weights = {'rating': 0.20, 'reviews': 0.10, 'sentiment': 0.10, 'followers': 0.15, 'mentions': 0.10, 'website': 0.05, 'style': 0.30}
    df['rating_norm'] = _safe_normalise(df['google_rating']); df['reviews_norm'] = _safe_normalise(df['review_count']); df['sentiment_norm'] = _safe_normalise(df['avg_sentiment_score']); df['followers_norm'] = _safe_normalise(df['instagram_followers']); df['mentions_norm'] = _safe_normalise(df['blog_mentions']); df['website_norm'] = df['has_website'].apply(lambda x: 1 if isinstance(x, str) and x.lower() == 'yes' else 0).fillna(0); df['uniqueness_norm'] = _safe_normalise(df['style_uniqueness_score'])
    df['composite_score'] = (df['rating_norm']*weights['rating'] + df['reviews_norm']*weights['reviews'] + df['sentiment_norm']*weights['sentiment'] + df['followers_norm']*weights['followers'] + df['mentions_norm']*weights['mentions'] + df['website_norm']*weights['website'] + df['uniqueness_norm']*weights['style'])
    return df

# --- NEW: Gemini API Fetching Logic ---
def get_live_metrics_for_venue(venue_name: str, venue_type: str) -> dict | None:
    """Uses Gemini with web search to find live metrics for a venue."""
    model = genai.GenerativeModel(model_name='gemini-1.5-pro-latest', tools=['google_search'])
    
    if venue_type == 'Restaurant':
        json_structure = """{"google_rating": float, "review_count": int, "price_range": int, "ambiance_score": int, "service_score": int, "uniqueness_score": int, "popularity_score": int, "hygiene_certification": int, "nri_friendly_score": int}"""
    else: # Boutique
        json_structure = """{"google_rating": float, "review_count": int, "avg_sentiment_score": float, "instagram_followers": int, "blog_mentions": int, "has_website": "Yes or No", "style_uniqueness_score": int}"""

    prompt = f"""
    Perform a targeted web search for the venue named "{venue_name}" in Trivandrum, India.
    Based on the search results, return a single, minified JSON object with the following metrics.
    Estimate scores on a 1-10 scale. Use `null` if a metric cannot be found.
    Provide ONLY the JSON object and no other text or formatting.
    {json_structure}
    """
    
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        metrics = json.loads(json_text)
        metrics['venue_name'] = venue_name
        return metrics
    except Exception as e:
        print(f"❌ ERROR processing '{venue_name}': {e}. Raw response: {response.text if 'response' in locals() else 'No response'}")
        return None

# --- Main Pipeline Logic ---
def run_the_pipeline():
    print("--- Starting LIVE Data Refresh Pipeline using Gemini API ---")
    base_venues_df = pd.read_csv(BASE_VENUES_PATH)
    
    restaurants_data = []
    boutiques_data = []
    
    for _, row in base_venues_df.iterrows():
        print(f"🔎 Researching: {row['venue_name']}...")
        metrics = get_live_metrics_for_venue(row['venue_name'], row['type'])
        if metrics:
            if row['type'] == 'Restaurant':
                restaurants_data.append(metrics)
            elif row['type'] == 'Boutique':
                boutiques_data.append(metrics)
        time.sleep(2) # Be kind to the API

    if not restaurants_data and not boutiques_data:
        print("Pipeline finished, but no new data was fetched.")
        return

    # --- Process and Score the fresh data ---
    restaurants_df = pd.DataFrame(restaurants_data)
    boutiques_df = pd.DataFrame(boutiques_data)
    
    restaurants_with_scores = calculate_restaurant_scores(restaurants_df)
    boutiques_with_scores = calculate_boutique_scores(boutiques_df)

    # --- Write final data to the database ---
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        restaurants_with_scores.to_sql('restaurants', conn, if_exists='replace', index=False)
        boutiques_with_scores.to_sql('boutiques', conn, if_exists='replace', index=False)
    finally:
        conn.close()
    
    print("\n✅ --- Pipeline finished. `recommender.db` has been updated with fresh data from the web! ---")

if __name__ == "__main__":
    run_the_pipeline()