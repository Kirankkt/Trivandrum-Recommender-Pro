import pandas as pd
import sqlite3
import os

# --- Configuration ---
# Define paths relative to this script's location for robustness
CURRENT_DIR = os.path.dirname(__file__)
# This points to your source data files
SOURCE_RESTAURANT_PATH = os.path.join(CURRENT_DIR, '..', 'source_data', 'restaurant_categorized_withcompscore.xlsx')
SOURCE_BOUTIQUE_PATH = os.path.join(CURRENT_DIR, '..', 'source_data', 'boutique_rankings.csv')
# This defines where the database will be created
DATABASE_DIR = os.path.join(CURRENT_DIR, '..', 'database')
DATABASE_PATH = os.path.join(DATABASE_DIR, 'recommender.db')

def run_the_pipeline():
    """
    Reads the source data files from the /source_data directory,
    and writes their contents into a new SQLite database file.
    """
    print("--- Starting Data Pipeline ---")

    # --- 1. Read Source Data ---
    print(f"Reading restaurant data from: {SOURCE_RESTAURANT_PATH}")
    restaurants_df = pd.read_excel(SOURCE_RESTAURANT_PATH)

    print(f"Reading boutique data from: {SOURCE_BOUTIQUE_PATH}")
    boutiques_df = pd.read_csv(SOURCE_BOUTIQUE_PATH)

    # --- 2. Create Database and Write Data ---
    print(f"Ensuring database directory exists at: {DATABASE_DIR}")
    os.makedirs(DATABASE_DIR, exist_ok=True)

    print(f"Connecting to database at: {DATABASE_PATH}")
    conn = sqlite3.connect(DATABASE_PATH)

    try:
        # The 'if_exists="replace"' option is key for updates.
        # It drops the old table and replaces it with the new data.
        print("Writing 'restaurants' table...")
        restaurants_df.to_sql('restaurants', conn, if_exists='replace', index=False)
        print("✅ Successfully wrote 'restaurants' table.")

        print("Writing 'boutiques' table...")
        boutiques_df.to_sql('boutiques', conn, if_exists='replace', index=False)
        print("✅ Successfully wrote 'boutiques' table.")

    except Exception as e:
        print(f"❌ An error occurred during database write: {e}")
    finally:
        conn.close()
        print("Database connection closed.")

    print("\n--- Data Pipeline Finished Successfully! ---")
    print(f"Database 'recommender.db' is now ready.")

if __name__ == "__main__":
    run_the_pipeline()