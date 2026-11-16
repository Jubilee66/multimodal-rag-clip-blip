import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="rag_docs"
)

query = input("Ask Question: ")

results = collection.query(
    query_texts=[query],
    n_results=5
)

for i in range(
    len(results["documents"][0])
):

    print("\n" + "="*60)
    print(f"RESULT {i+1}")
    print("="*60)

    print(
        results["documents"][0][i][:1000]
    )

    print("\nSOURCE:")
    print(
        results["metadatas"][0][i]
    )