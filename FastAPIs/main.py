from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import httpx
import numpy as np
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Config
# -----------------------------
MARTELLO_API = "https://martello.onrender.com/api/products"   # adjust to your Martello backend

# Load BERT once
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request Body
# -----------------------------
class SearchRequest(BaseModel):
    query: str

# -----------------------------
# Helpers
# -----------------------------
async def fetch_products():
    """Fetch all products from Martello API"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(MARTELLO_API)
        resp.raise_for_status()
        return resp.json()  # should return list of products


def keyword_filter(products, query_terms: List[str]):
    return [
        p for p in products
        if any(term in p["name"].lower() for term in query_terms)
    ]


def semantic_search(products, query_terms: List[str], top_k=3):
    names = [p["name"] for p in products]
    product_embeddings = embedder.encode(names, convert_to_tensor=True)
    results = set()

    for term in query_terms:
        q_emb = embedder.encode(term, convert_to_tensor=True)
        cos_scores = util.cos_sim(q_emb, product_embeddings)[0]
        top_results = np.argsort(-cos_scores.cpu().numpy())[:top_k]
        for idx in top_results:
            results.add(idx)

    return [products[i] for i in results]

# -----------------------------
# POST Endpoint
# -----------------------------
@app.post("/search-products")
async def search_products(body: SearchRequest):
    products = await fetch_products()
    q = body.query.strip()

    if not q:
        return {"products": products}

    query_terms = [term.strip().lower() for term in q.split(",") if term.strip()]

    keyword_results = keyword_filter(products, query_terms)

    if not keyword_results:
        semantic_results = semantic_search(products, query_terms)
        return {"products": semantic_results}

    semantic_results = semantic_search(products, query_terms)
    merged = {p["_id"]: p for p in keyword_results + semantic_results}

    return {"products": list(merged.values())}