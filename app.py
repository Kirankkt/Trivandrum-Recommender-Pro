from __future__ import annotations
import os
import sqlite3
import faiss, folium, gradio as gr, numpy as np, pandas as pd
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# --- GLOBAL CONFIG (Unchanged) ---
REST_METRICS = {"Rating": "Avg_Rating", "Ambiance": "Ambiance_Score", "Service": "Service_Score", "Uniqueness": "Uniqueness_Score", "Popularity": "Popularity_Score", "Price (cheap)": "Price_Range"}
BOUT_METRICS = {"Rating": "Google Rating", "Reviews": "Review Count", "Followers": "Instagram Followers", "Style": "Style Uniqueness Score"}

class TrivandrumAnalytics:
    def __init__(self):
        print("🚀  Initialising Trivandrum Analytics Engine…")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.db_path = os.path.join(os.path.dirname(__file__), 'database', 'recommender.db')
        self._load_data_from_db()
        self._create_embeddings()
        self._prepare_similarity_features()
        print("✅  Engine setup complete.")

    def _load_data_from_db(self):
        print(f"Reading data from SQLite database: {self.db_path}")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at '{self.db_path}'. Please run `python data_pipeline/run_pipeline.py` first to create it.")
        
        conn = sqlite3.connect(self.db_path)
        try:
            rest_raw = pd.read_sql('SELECT * FROM restaurants', conn)
            id_vars = [c for c in rest_raw.columns if c not in ['Restaurant_Name', 'Hotels', 'Café']]
            rest = pd.melt(rest_raw, id_vars=id_vars, value_vars=['Restaurant_Name', 'Hotels', 'Café'], var_name='venue_type', value_name='venue_name').dropna(subset=['venue_name']).reset_index(drop=True)
            rest['venue_type'] = rest['venue_type'].map({'Restaurant_Name': 'Restaurant', 'Hotels': 'Hotel', 'Café': 'Café'})
            for c in REST_METRICS.values():
                if c in rest.columns: rest[f"{c}_norm"] = self._safe_normalise(rest[c])
            self.restaurant_df = rest

            bout = pd.read_sql('SELECT * FROM boutiques', conn)
            bout = bout.rename(columns={"Boutique Name": "venue_name"})
            for c in BOUT_METRICS.values():
                if c in bout.columns: bout[c] = pd.to_numeric(bout[c], errors="coerce"); bout[f"{c}_norm"] = self._safe_normalise(bout[c])
            self.boutique_df = bout
        finally:
            conn.close()
    
    # ... The rest of the TrivandrumAnalytics class methods are exactly the same as the last stable version ...
    def _safe_normalise(self, series: pd.Series) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(series): return pd.Series([0.5] * len(series), index=series.index)
        lo, hi = series.min(), series.max();
        if hi == lo: return pd.Series([0.5] * len(series), index=series.index)
        return (series - lo) / (hi - lo)
    def _create_embeddings(self):
        # ... same embedding code ...
        pass
    # ... all other methods ...

# (For brevity, I've omitted the full class code, but you should paste the entire stable `TrivandrumAnalytics` class and the full `gr.Blocks` UI code from the last version you confirmed was working well.)

# Placeholder for the full app code from your last working version
engine = TrivandrumAnalytics()
with gr.Blocks() as demo:
    gr.Markdown("# Trivandrum Recommender (Dynamic Version)")
    gr.Markdown("This app is now reading from a SQLite database.")
    # Paste your FULL Gradio UI layout and callbacks here.

if __name__ == "__main__":
    demo.launch()