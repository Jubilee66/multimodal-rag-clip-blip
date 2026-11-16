#---uses scripts in C:\Education\Code\AgenticAI\PRACTICE_SELF\RAG\RAG_1\test_RAG(singleDoc)\utils------
from pathlib import Path
from utils.pdf_loader import load_pdf
from utils.docx_loader import load_docx
from utils.excel_loader import load_excel
from utils.image_loader import load_image

DATA_DIR = Path(
    r"C:\Education\Code\AgenticAI\PRACTICE_SELF\RAG\RAG_2_MultimodalRAG\data")

all_documents = []

for file in DATA_DIR.iterdir():

    print(f"\nLoading: {file.name}")

    try:

        suffix = file.suffix.lower()

        if suffix == ".pdf":
            text = load_pdf(file)

        elif suffix == ".docx":
            text = load_docx(file)

        elif suffix in [".xlsx", ".xls"]:
            text = load_excel(file)

        elif suffix in [".png", ".jpg", ".jpeg"]:
            text = load_image(file)

        else:
            continue

        if not text.strip():
            print(f"No text extracted from {file.name}")
            continue

        all_documents.append(
            {
                "source": file.name,
                "type": suffix,
                "content": text
            }
        )

        print(f"Characters Extracted: {len(text)}")

    except Exception as e:

        print(
            f"Error processing {file.name}: {e}"
        )

print("\n" + "=" * 60)
print(f"Loaded Documents: {len(all_documents)}")
print("=" * 60)

for doc in all_documents:
    print(
        f"{doc['source']} -> {len(doc['content'])} chars"
    )







    # from utils.pdf_loader import load_pdf



