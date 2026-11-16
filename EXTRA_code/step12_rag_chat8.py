import os
import chromadb
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv(
        "GOOGLE_API_KEY"
    )
)

client = chromadb.PersistentClient(
    path="./chroma_db_v2"
)

collection = client.get_collection(
    "rag_docs"
)

model = genai.GenerativeModel(
    "gemini-1.5-flash"
)

while True:

    question = input(
        "\nAsk: "
    )

    if question.lower() == "exit":
        break

    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    context = "\n".join(
        results["documents"][0]
    )

    prompt = f"""
Answer using the context.

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(
        prompt
    )

    print("\n")
    print(response.text)