import chromadb

# TEXT DB
text_client = chromadb.PersistentClient(
    path="./chroma_db_txt"
)

text_collection = text_client.get_collection(
    "rag_docs"
)

# IMAGE DB
image_client = chromadb.PersistentClient(
    path="./chroma_db_img"
)

image_collection = image_client.get_collection(
    "image_collection"
)