import os
import logging
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch



# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

TEXT_INDEX = "furniture-text"
IMAGE_INDEX = "furniture-image"



# Load embeddings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "product_embeddings.parquet")

df = pd.read_parquet(DATA_PATH)



# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

if TEXT_INDEX not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=TEXT_INDEX,
        dimension=len(df["text_embedding"][0]),
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

if IMAGE_INDEX not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=IMAGE_INDEX,
        dimension=len(df["image_embedding"][0]),
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

text_index = pc.Index(TEXT_INDEX)
image_index = pc.Index(IMAGE_INDEX)



# Upsert embeddings (run once)
def upsert_embeddings():
    text_vectors = []
    image_vectors = []

    for _, row in df.iterrows():
        text_vectors.append({
            "id": f"text_{row['uniq_id']}",
            "values": row['text_embedding'],
            "metadata": {
                "type": "text",
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
        image_vectors.append({
            "id": f"image_{row['uniq_id']}",
            "values": row['image_embedding'],
            "metadata": {
                "type": "image",
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

    batch_size = 100
    for i in range(0, len(text_vectors), batch_size):
        text_index.upsert(vectors=text_vectors[i:i + batch_size])
    for i in range(0, len(image_vectors), batch_size):
        image_index.upsert(vectors=image_vectors[i:i + batch_size])

    print(f"✅ Uploaded {len(text_vectors)} text vectors and {len(image_vectors)} image vectors.")

# uncomment and run only for the first time
# upsert_embeddings()



# Load or Download Models
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)



# Text Embedding Model
TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "text_model")

if os.path.exists(TEXT_MODEL_PATH):
    embedding_model = SentenceTransformer(TEXT_MODEL_PATH)
    print("✅ Loaded local text model.")



# Image Embedding Model
IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "image_model.pth")

weights = ViT_B_16_Weights.DEFAULT
image_model = vit_b_16(weights=None)

if os.path.exists(IMAGE_MODEL_PATH):
    image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location="cpu"))
    print("✅ Loaded local image model.")

image_model.eval()


def get_text_embedding(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()


def get_image_embedding(file_bytes: bytes) -> List[float]:
    if image_model is None:
        return [0.0] * 1000
    from PIL import Image
    from io import BytesIO
    transform = ViT_B_16_Weights.DEFAULT.transforms()
    try:
        img = Image.open(BytesIO(file_bytes)).convert('RGB')
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            emb = image_model(img_t)
        return emb.squeeze().numpy().tolist()
    except Exception:
        return [0.0] * 1000


# FastAPI app
app = FastAPI(title="Furniture Recommendation API")

TEXT_DIM = 384
IMAGE_DIM = 1000


@app.post("/recommend")
async def recommend(
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    top_k: int = Form(5)
):
    if query and image:
        raise HTTPException(status_code=400, detail="Please provide either text or an image, not both.")

    if query:
        query_vector = get_text_embedding(query)
        response = text_index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        vector_type = "text"
    elif image and image.filename:
        image_bytes = await image.read()
        query_vector = get_image_embedding(image_bytes)
        response = image_index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        vector_type = "image"
    else:
        return {"matches": []}

    matches: List[Dict[str, Any]] = []
    for match in getattr(response, "matches", []) or []:
        metadata = getattr(match, "metadata", {}) if not isinstance(match, dict) else match.get("metadata", {})
        if metadata.get("type") == vector_type:
            matches.append(metadata)

    return {"matches": matches}


# Logging & CORS
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


# resolve port issue in Render (comment out for local testing)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
