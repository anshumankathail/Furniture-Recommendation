import os
import pandas as pd
from dotenv import load_dotenv

import logging
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, cast, Optional

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer



# Load environment variables
load_dotenv()  # from .env in project root
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "furniture-recommendation"



# Load embeddings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "product_embeddings.parquet")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Embedding file not found: {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)



# Initialize Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=len(df["embedding"][0]),
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # adjust region if needed
    )



# Connect to the index
index = pc.Index(INDEX_NAME)



# Upsert embeddings (run once)
def upsert_embeddings():
    vectors = []
    for _, row in df.iterrows():
        vectors.append({
            "id": row["uniq_id"],
            "values": row["embedding"],
            "metadata": {
                "title": row["title"],
                "brand": row["brand"],
                "color": row.get("color", "Unknown"),
                "material": row.get("material", "Unknown"),
                "price": row.get("price", "Unknown"),
                "country_of_origin": row.get("country_of_origin", "Unknown"),
                "package_dimensions": row.get("package_dimensions", "Unknown"),
                "image_url": row.get("image_url", "")
            }
        })
    # Upsert in batches of 100 (optional)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
    print(f"✅ Uploaded {len(vectors)} vectors to Pinecone.")



# -----------------------------------------------------------------
# Uncomment the line below only the first time you run this backend
# upsert_embeddings()
# -----------------------------------------------------------------



# Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_text_embedding(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()



# FastAPI app
app = FastAPI(title="Furniture Recommendation API")

TEXT_DIM = 384  # Dimension of text embedding
IMAGE_DIM = 1000  # Dimension of image embedding
TOTAL_DIM = TEXT_DIM + IMAGE_DIM

def get_image_embedding(file_bytes: bytes) -> List[float]:
    return [0.0] * IMAGE_DIM

class QueryItem(BaseModel):
    query: Optional[str] = None
    top_k: int = 5

@app.post("/recommend")
async def recommend(
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    top_k: int = Form(5)
):
    # ❌ Prevent both being sent together
    if query and image:
        raise HTTPException(status_code=400, detail="Please provide either text or an image, not both.")

    # 1️⃣ Text embedding
    text_embedding = get_text_embedding(query) if query else [0.0] * TEXT_DIM

    # 2️⃣ Image embedding
    if image and image.filename:
        image_bytes = await image.read()
        img_embedding = get_image_embedding(image_bytes)
    else:
        img_embedding = [0.0] * IMAGE_DIM

    # 3️⃣ Combine embeddings
    query_vector = text_embedding + img_embedding

    # 4️⃣ Query Pinecone
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # 5️⃣ Extract matches safely
    matches: List[Dict[str, Any]] = []
    matches_raw = getattr(response, "matches", []) or []

    for match in matches_raw:
        metadata = getattr(match, "metadata", {}) if not isinstance(match, dict) else match.get("metadata", {})
        matches.append(metadata)

    return {"matches": matches}



# detailed error logs
logging.basicConfig(level=logging.INFO)

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    logging.error(f"Error: {exc}", exc_info=True)
    return PlainTextResponse(str(exc), status_code=500)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)