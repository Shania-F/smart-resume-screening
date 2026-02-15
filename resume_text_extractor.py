import os
import fitz  # PyMuPDF
from PIL import Image
import io
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import pytesseract

# tesseract not in PATH
pytesseract.pytesseract.tesseract_cmd = r"D:\PycharmProjects\cv_processor\Tesseract-OCR\tesseract.exe"


def extract_text_from_image_pdf(pdf_path):
    """OCR all pages of a PDF using PyMuPDF + Tesseract."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc, start=1):
        # render page to image
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        page_text = pytesseract.image_to_string(img)
        text += page_text
    return text

def load_documents(folder_path):
    docs = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        print(f"Processing file: {file}")

        if file.endswith(".pdf"):
            # Try normal PDF first
            loader = PyPDFLoader(path)
            documents = loader.load()
            if not any(doc.page_content.strip() for doc in documents):
                # fallback to OCR if PDF is scanned
                print(f"Triggering OCR fallback for {file}")
                ocr_text = extract_text_from_image_pdf(path)
                documents = [Document(page_content=ocr_text, metadata={"source": file})]

        # elif file.endswith(".docx"):
        #     loader = Docx2txtLoader(path)
        #     documents = loader.load()

        else:
            continue

        # add filename metadata
        for doc in documents:
            doc.metadata["source"] = file

        # some extracted chunks may be too small to have any meaningful info
        # documents = [doc for doc in documents if len(doc.page_content.strip()) > 100]
        docs.extend(documents)

    return docs


if __name__ == "__main__":
    folder_path = "resumes"
    documents = load_documents(folder_path)

    print(f"\nTotal documents extracted: {len(documents)}\n")

    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        snippet = doc.page_content[:200].replace("\n", " ")  # first 200 chars
        char_count = len(doc.page_content)

        print(f"Document {i}: {source}")
        print(f" - Characters extracted: {char_count}")
        print(f" - Preview: {snippet}...")
        print("-" * 50)


# other preprocessing steps
# Remove headers/footers, images, unnecessary formatting.
# Normalize whitespace, lowercase, remove stopwords if desired.
# Extract relevant sections (optional): experience, skills, education.
