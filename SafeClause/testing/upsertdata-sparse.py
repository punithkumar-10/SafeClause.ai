import os
import json
from pinecone import Pinecone

# --- CONFIGURATION ---
PINECONE_API_KEY = "pcsk_2wFprz_FWmjj2zwhCuAZNqheBR4mtP8FU7VUggLqQQUwZhJaWFMcCK2NXaSC5h26LZzVap"
INDEX_HOST = "https://safeclause-sparse-dc5puwa.svc.aped-4627-b74a.pinecone.io"
NAMESPACE = "Acts_and_Clause_Namespace"
CHUNKS_DIR = "SafeClause/Chunks"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=INDEX_HOST)

def upsert_all_files():
    # Loop through each file in the directory
    for filename in os.listdir(CHUNKS_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(CHUNKS_DIR, filename)
            print(f"Processing: {filename}")

            # 1. Load Data
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 2. Upsert Data
            # Passing the data list directly as requested
            index.upsert_records(NAMESPACE, data)
            
            print(f"   ✅ Upserted {filename}")

if __name__ == "__main__":
    upsert_all_files()