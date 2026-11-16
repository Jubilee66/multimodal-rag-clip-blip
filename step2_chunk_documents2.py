from langchain_text_splitters import RecursiveCharacterTextSplitter
# from RAG.RAG_2_MultimodalRAG.step1_ingest_all1 import all_documents
from step1_ingest_all1 import all_documents

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_chunks = []

for doc in all_documents:

    chunks = splitter.split_text(doc["content"])

    for chunk in chunks:
        all_chunks.append({
            "source": doc["source"],
            "type": doc["type"],
            "content": chunk
        })

print(f"\nTotal Chunks: {len(all_chunks)}")

for chunk in all_chunks[:3]:
    print("\n", "="*50)
    print(chunk["source"])
    print(chunk["content"][:500])