from docx import Document

def load_docx(file):

    doc = Document(file)

    text = "\n".join(
        para.text
        for para in doc.paragraphs
    )

    return text