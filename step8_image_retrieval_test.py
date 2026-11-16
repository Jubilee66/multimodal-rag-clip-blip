# =====================================================
# IMAGE SEARCH ENGINE
#
# User Query
#      ↓
#
# CLIP Text Encoder
#      ↓
#
# Search image_collection
#      ↓
#
# Return matching images + captions
#
# =====================================================

import chromadb

from sentence_transformers import (
    SentenceTransformer
)

# -----------------------------------------------------
# CLIP MODEL
# -----------------------------------------------------

model = SentenceTransformer(
    "clip-ViT-B-32"
)

# -----------------------------------------------------
# IMAGE DATABASE
# -----------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db_img"
)

collection = client.get_collection(
    "image_collection"
)

# -----------------------------------------------------
# QUERY
# -----------------------------------------------------

query = input(
    "Ask: "
)

query_embedding = model.encode(
    query
)

# -----------------------------------------------------
# SEARCH
# -----------------------------------------------------

results = collection.query(
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=5
)

# -----------------------------------------------------
# DISPLAY RESULTS
# -----------------------------------------------------

print("\n" + "=" * 60)
print("IMAGE SEARCH RESULTS")
print("=" * 60)

for i, meta in enumerate(
    results["metadatas"][0],
    start=1
):

    print(f"\nIMAGE {i}")
    print("-" * 40)

    print(
        f"File: {meta['image']}"
    )

    print(
        f"Caption: "
        f"{meta.get('caption', 'No caption available')}"
    )

print("\nSearch Completed.")