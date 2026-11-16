#--------------------------------------------------------
#--------------------------------------------------------
#--------------ARCHITECTURE NEW--------------------------
#------------------SHORT--------------------------------------
# PDF/DOCX/Excel/Image → OCR/Parsing → Chunking → MiniLM Embeddings → ChromaDB → Retrieval → Llama 3.2 → Streamlit
#------------------DETAIL--------------------------------------
# OCR Model
# ↓
# Tesseract OCR

# Embedding Model
# ↓
# all-MiniLM-L6-v2

# Vector Database
# ↓
# ChromaDB

# Generative LLM
# ↓
# Llama 3.2 (via Ollama)

# Frontend
# ↓
# Streamlit
#--------------------------------------------------------
#--------------------------------------------------------



import chromadb
import ollama

client = chromadb.PersistentClient(
    path="./chroma_db_txt"
)

collection = client.get_collection(
    "rag_docs"
)

def ask_rag(question):

    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    context = "\n\n".join(
        results["documents"][0]
    )

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]



#--------------------------------------------------------
#--------------------------------------------------------
#---------------ARCHITECTURE OLD-------------------------
#--------------------SHORT------------------------------------
# Documents → Chunking → MiniLM Embeddings → ChromaDB → Retrieval → Streamlit
#--------------------DETAIL------------------------------------
# PDF/DOCX/Excel/Image
# → Text Extraction (PyPDF, python-docx, pandas, Tesseract OCR)
# → Chunking
# → all-MiniLM-L6-v2 Embeddings
# → ChromaDB Vector Store
# → Similarity Search (Top-K)
# → Retrieved Chunks
# → Streamlit UI
#--------------------------------------------------------
#--------------------------------------------------------

# import chromadb

# client = chromadb.PersistentClient(
#     path="./chroma_db_v2"
# )

# collection = client.get_collection(
#     "rag_docs"
# )
# #-----------Simple ask-----------------
# # query = input("Ask: ")

# # results = collection.query(
# #     query_texts=[query],
# #     n_results=3
# # )

# # for doc in results["documents"][0]:
# #     print("\n")
# #     print(doc[:1000])

# #------------Ask as per app.py------------

# def ask_rag(question):

#     results = collection.query(
#         query_texts=[question],
#         n_results=3
#     )

#     context = "\n".join(
#         results["documents"][0]
#     )

#     return context


