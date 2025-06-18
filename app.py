from __future__ import annotations
import os
import sqlite3
import faiss, folium, gradio as gr, numpy as np, pandas as pd
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

REST_METRICS = {"Rating": "avg_rating", "Ambiance": "ambiance_score", "Service": "service_score", "Uniqueness": "uniqueness_score", "Popularity": "popularity_score", "Price (cheap)": "price_range"}
BOUT_METRICS = {"Rating": "google rating", "Reviews": "review count", "Followers": "instagram followers", "Style": "style uniqueness score"}
STOP_WORDS = {"best", "good", "nice", "top", "great", "restaurant", "restaurants", "cafe", "cafes", "coffee", "hotel", "hotels", "boutique", "boutiques", "place", "places", "near", "me", "in", "around", "of", "cuisine", "food", "eat", "dine"}

def _safe_normalise(series: pd.Series) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors='coerce').fillna(0)
    lo, hi = numeric_series.min(), numeric_series.max()
    if hi == lo: return pd.Series(0.5, index=series.index)
    return (numeric_series - lo) / (hi - lo)

def _load_or_build_embeddings(model, texts, cache_path):
    cache = os.path.join(os.path.dirname(__file__), cache_path)
    if os.path.exists(cache): emb = np.load(cache)
    else: emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True); os.makedirs(os.path.dirname(cache), exist_ok=True); np.save(cache, emb)
    return emb.astype("float32")

def _haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0; lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2]); dlat, dlon = lat2 - lat1, lon2 - lon1; a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2; return 2 * R * np.arcsin(np.sqrt(a))

def _folium_html(df, lat_col="latitude", lon_col="longitude", name_col="venue_name", cat="Restaurants"):
    df = df.dropna(subset=[lat_col, lon_col])
    if df.empty: return "<p>No coordinates yet.</p>"
    center = [df[lat_col].mean(), df[lon_col].mean()]; m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron"); icon_color = "blue" if cat == "Restaurants" else "green"; icon_symbol = "cutlery" if cat == "Restaurants" else "shopping-cart"
    for _, row in df.iterrows(): folium.Marker((row[lat_col], row[lon_col]), popup=row[name_col], icon=folium.Icon(color=icon_color, icon=icon_symbol, prefix="fa")).add_to(m)
    return m._repr_html_()

def _clean_query(q: str) -> str:
    tokens = [t for t in q.lower().split() if t not in STOP_WORDS]; cleaned = " ".join(tokens)
    if not cleaned.strip():
        if "boutique" in q.lower(): return "boutique"
        if any(k in q.lower() for k in ["restaurant", "cafe", "hotel"]): return "restaurant"
    return cleaned

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
            # Load and process restaurants
            rest_raw = pd.read_sql('SELECT * FROM restaurants', conn)
            id_vars = [c for c in rest_raw.columns if c not in ['Restaurant_Name', 'Hotels', 'Café']]
            rest = pd.melt(rest_raw, id_vars=id_vars, value_vars=['Restaurant_Name', 'Hotels', 'Café'], var_name='venue_type', value_name='venue_name').dropna(subset=['venue_name']).reset_index(drop=True)
            rest['venue_type'] = rest['venue_type'].map({'Restaurant_Name': 'Restaurant', 'Hotels': 'Hotel', 'Café': 'Café'})
            self.restaurant_df = rest.rename(columns=str.lower)
            
            # Load and process boutiques
            bout = pd.read_sql('SELECT * FROM boutiques', conn)
            self.boutique_df = bout.rename(columns=str.lower).rename(columns={"boutique name": "venue_name"})
        finally:
            conn.close()

    def _create_embeddings(self):
        print("Creating text embeddings...")
        r_text = [f"{r['venue_name']} rating {r.get('avg_rating',0):.1f}" for _, r in self.restaurant_df.iterrows()]
        b_text = [f"{r['venue_name']} style {r.get('style uniqueness score',0)}" for _, r in self.boutique_df.iterrows()]
        r_emb = _load_or_build_embeddings(self.model, r_text, "data/rest_emb.npy"); b_emb = _load_or_build_embeddings(self.model, b_text, "data/bout_emb.npy")
        self.r_index = faiss.IndexFlatL2(r_emb.shape[1]); self.r_index.add(r_emb)
        self.b_index = faiss.IndexFlatL2(b_emb.shape[1]); self.b_index.add(b_emb)

    def _prepare_similarity_features(self):
        print("Preparing similarity features...")
        rest_features = list(REST_METRICS.values()); self.restaurant_features = self.restaurant_df[rest_features].fillna(0)
        bout_features = list(BOUT_METRICS.values()); self.boutique_features = self.boutique_df[bout_features].fillna(0)
        self.rest_scaler = StandardScaler(); self.bout_scaler = StandardScaler()
        self.restaurant_features_scaled = self.rest_scaler.fit_transform(self.restaurant_features)
        self.boutique_features_scaled = self.bout_scaler.fit_transform(self.boutique_features)

    def _get_df_and_cols(self, cat, sub_cat):
        if cat == "Restaurants": df, name_col, p_cat = self.restaurant_df, 'venue_name', 'Restaurants'
        else: df, name_col, p_cat = self.boutique_df, 'venue_name', 'Boutiques'
        if cat == "Restaurants" and sub_cat != "All": df = df[df['venue_type'] == sub_cat]
        return df.copy(), name_col, p_cat
    
    # ... rest of the class methods ...
    def find_similar(self, name, pretty_cat, sub_cat):
        if not name: return "Please select a place first."
        # ... logic ...
        return "Find similar is working!"


engine = TrivandrumAnalytics()

with gr.Blocks(theme=gr.themes.Soft(), css="footer{display:none !important}") as demo:
    gr.Markdown("# ⭐ Trivandrum Recommendation Assistant (Dynamic)")
    RESTAURANT_SUB_CATS = ["All", "Restaurant", "Hotel", "Café"]
    with gr.Tabs(selected=0):
        with gr.TabItem("🏆 Leaderboard"):
            with gr.Row():
                with gr.Column(scale=1): l_cat = gr.Radio(["Restaurants", "Boutiques"], value="Restaurants", label="Category"); l_sub_cat = gr.Radio(RESTAURANT_SUB_CATS, value="All", label="Type", interactive=True); l_metric = gr.Dropdown(choices=["Composite"] + list(REST_METRICS), value="Composite", label="Metric")
                with gr.Column(scale=2): l_plot, l_table = gr.Plot(), gr.Markdown()
    
    # --- CALLBACKS ---
    def _render_leader(cat, sub_cat, metric_label):
        df, name_col, pretty_cat = engine._get_df_and_cols(cat, sub_cat)
        # ... full callback logic here ...
        return go.Figure(), "Leaderboard will be rendered here."

    l_metric.change(_render_leader, [l_cat, l_sub_cat, l_metric], [l_plot, l_table], queue=False)
    # ... other callbacks ...


if __name__ == "__main__":
    demo.launch()