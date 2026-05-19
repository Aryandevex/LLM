# Hybrid Search = BM25 + Vector Similarity
# Simple Example

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# Documents
# -----------------------------
documents = [
    "Python is used for AI and machine learning",
    "MongoDB is a NoSQL database",
    "BM25 is a keyword search algorithm",
    "Vector search uses embeddings",
    "Hybrid search combines BM25 and vector search"
]

# -----------------------------
# BM25 Setup
# -----------------------------
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# -----------------------------
# Vector Embedding Setup
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents)

# -----------------------------
# Query
# -----------------------------
query = "search using embeddings and keywords"

# BM25 Scores
tokenized_query = query.lower().split()
bm25_scores = bm25.get_scores(tokenized_query)

# Vector Scores
query_embedding = model.encode([query])
vector_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

# -----------------------------
# Hybrid Score
# -----------------------------
# Normalize scores
bm25_scores = bm25_scores / np.max(bm25_scores)
vector_scores = vector_scores / np.max(vector_scores)

# Combine scores
alpha = 0.5   # weight for BM25
beta = 0.5    # weight for Vector

hybrid_scores = (alpha * bm25_scores) + (beta * vector_scores)

# -----------------------------
# Ranking
# -----------------------------
ranked_results = sorted(
    zip(documents, hybrid_scores),
    key=lambda x: x[1],
    reverse=True
)

# -----------------------------
# Output
# -----------------------------
print("\nTop Results:\n")

for doc, score in ranked_results:
    print(f"Score: {score:.4f}")
    print(f"Doc: {doc}")
    print("-" * 50)
