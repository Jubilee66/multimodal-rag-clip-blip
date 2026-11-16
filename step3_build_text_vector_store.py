# --------------------------------------------------
# STORE CHUNKS WITH MiniLM EMBEDDINGS
# Chunks
#   ↓
# MiniLM Embeddings
#   ↓
# ChromaDB
# --------------------------------------------------

import chromadb
from sentence_transformers import SentenceTransformer
from step2_chunk_documents2 import all_chunks

# Embedding Model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db_txt"
)

collection = client.get_or_create_collection(
    name="rag_docs"
)

for i, chunk in enumerate(all_chunks):

    embedding = model.encode(
        chunk["content"]
    ).tolist()

    collection.add(
        ids=[f"chunk_{i}"],
        documents=[chunk["content"]],
        embeddings=[embedding],
        metadatas=[
            {
                "source": chunk["source"],
                "type": chunk["type"]
            }
        ]
    )

print(f"Stored {len(all_chunks)} chunks successfully.")
#-------------------------------------------------------
# import chromadb
# from sentence_transformers import SentenceTransformer
# # from RAG.RAG_2_MultimodalRAG.step2_chunk_documents2 import all_chunks
# from step2_chunk_documents2 import all_chunks
# model = SentenceTransformer(
#     "all-MiniLM-L6-v2"
# )

# client = chromadb.PersistentClient(
#     path="./chroma_db_v3"
# )

# collection = client.get_or_create_collection(
#     name="rag_docs"
# )

# for i, chunk in enumerate(all_chunks):

#     embedding = model.encode(
#         chunk["content"]
#     ).tolist()

#     collection.add(
#         ids=[str(i)],
#         documents=[chunk["content"]],
#         embeddings=[embedding],
#         metadatas=[
#             {
#                 "source": chunk["source"]
#             }
#         ]
#     )

# print("Done")