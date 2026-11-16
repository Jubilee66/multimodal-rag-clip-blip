# =====================================================
# MULTIMODAL HYBRID RAG
#
# Question
#  ↓

# Text Retrieval
# +
# Image Retrieval

#  ↓

# Vision Captioning

#  ↓

# Combined Context

#  ↓

# Llama 3.2

#  ↓

# Answer + Images
# =====================================================
# CUURENT:
# Question
#  ↓
# CLIP Retrieval
#  ↓
# Retrieved Images
#  ↓
# BLIP
#  ↓
# Image Caption
#  ↓
# LLM
# =======================================================
import chromadb
import ollama

from sentence_transformers import SentenceTransformer

# =====================================================
# MODELS
# =====================================================

text_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

image_model = SentenceTransformer(
    "clip-ViT-B-32"
)

# =====================================================
# DATABASES
# =====================================================

text_client = chromadb.PersistentClient(
    path="./chroma_db_txt"
)

image_client = chromadb.PersistentClient(
    path="./chroma_db_img"
)

text_collection = text_client.get_collection(
    "rag_docs"
)

image_collection = image_client.get_collection(
    "image_collection"
)

# =====================================================
# HYBRID SEARCH
# =====================================================

def hybrid_search(query):

    # -------------------------
    # TEXT SEARCH
    # -------------------------

    text_embedding = text_model.encode(query)

    text_results = text_collection.query(
        query_embeddings=[
            text_embedding.tolist()
        ],
        n_results=3
    )

    # -------------------------
    # IMAGE SEARCH
    # -------------------------

    image_embedding = image_model.encode(
        query
    )

    image_results = image_collection.query(
        query_embeddings=[
            image_embedding.tolist()
        ],
        n_results=3
    )

    return text_results, image_results


# =====================================================
# RAG FUNCTION FOR APP.PY
# =====================================================

# def ask_rag(question):

#     text_results, image_results = hybrid_search(
#         question
#     )

#     docs = text_results["documents"][0]

#     text_context = "\n\n".join(
#         docs
#     )

#     image_names = []

#     for meta in image_results[
#         "metadatas"
#     ][0]:

#         image_names.append(
#             meta["image"]
#         )

#     image_context = "\n".join(
#         image_names
#     )

#     prompt = f"""
# You are a Multimodal RAG Assistant.

# Answer using ONLY the retrieved context.

# TEXT CONTEXT:
# {text_context}

# RELATED IMAGES:
# {image_context}

# QUESTION:
# {question}

# ANSWER:
# """

#     response = ollama.chat(
#         model="llama3.2",
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )

#     return response[
#         "message"
#     ]["content"]
def ask_rag(question):
    
    # ----------------------------------
    # RETRIEVE TEXT + IMAGES
    # ----------------------------------

    text_results, image_results = hybrid_search(
        question
    )

    # ----------------------------------
    # TEXT CONTEXT
    # ----------------------------------

    docs = text_results["documents"][0]

    text_context = "\n\n".join(
        docs
    )

    # ----------------------------------
    # IMAGE INFO FOR STREAMLIT DISPLAY
    # ----------------------------------

    image_names_info = []

    for meta in image_results["metadatas"][0]:

        image_names_info.append(
            {
                "name": meta["image"],
                "path": f"data/extracted_images/{meta['image']}"
            }
        )

    # ----------------------------------
    # IMAGE CONTEXT FOR LLM
    # ----------------------------------

    image_context = ""

    for meta in image_results["metadatas"][0]:

        image_context += f"""
IMAGE:
{meta['image']}

CAPTION:
{meta.get('caption', 'No caption available')}

"""

    # ----------------------------------
    # PROMPT
    # ----------------------------------

    prompt = f"""
You are a Multimodal RAG Assistant.

Answer ONLY using the retrieved context.

TEXT CONTEXT:
{text_context}

IMAGE CONTEXT:
{image_context}

QUESTION:
{question}

ANSWER:
"""

    # ----------------------------------
    # LLM
    # ----------------------------------

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # ----------------------------------
    # RETURN
    # ----------------------------------

    return {
        "answer": response["message"]["content"],
        "images": image_names_info
    }
# =====================================================
# TERMINAL TEST MODE
# =====================================================

if __name__ == "__main__":

    while True:

        query = input(
            "\nAsk (type exit): "
        )

        if query.lower() == "exit":
            break

        text_results, image_results = hybrid_search(
            query
        )

        print("\n" + "=" * 60)
        print("TEXT RESULTS")
        print("=" * 60)

        docs = text_results[
            "documents"
        ][0]

        for i, doc in enumerate(
            docs,
            start=1
        ):

            print(f"\nChunk {i}")
            print("-" * 40)
            print(doc[:500])

        print("\n" + "=" * 60)
        print("IMAGE RESULTS")
        print("=" * 60)

        image_metas = image_results[
            "metadatas"
        ][0]

        for i, meta in enumerate(
            image_metas,
            start=1
        ):
            print(
                f"\nImage {i}: "
                f"{meta['image']}"
            )

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)

        answer = ask_rag(
            query
        )

        # print(answer)
        print(answer["answer"])

        print("\nRetrieved Images:")

        for img in answer["images"]:
            print(img)