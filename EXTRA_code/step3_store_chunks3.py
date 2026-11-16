#-------------store chunk-------------------------------
# Chunks
#  ↓
# Store in Chroma
#---------------------------------------------

import chromadb

# from RAG.RAG_2_MultimodalRAG.step2_chunk_documents2 import all_chunks
from step2_chunk_documents2 import all_chunks
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="rag_docs"
)

for i, chunk in enumerate(all_chunks):

    collection.add(
        ids=[str(i)],
        documents=[chunk["content"]],
        metadatas=[
            {
                "source": chunk["source"],
                "type": chunk["type"]
            }
        ]
    )

print(f"\nStored {len(all_chunks)} chunks successfully")

# python "C:\Education\Code\AgenticAI\PRACTICE_SELF\RAG\RAG_2_MultimodalRAG\step4_Better_Embeddings6.py"