import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- CONFIG ---
PDF_FOLDER = "KB_web_docs"
VECTOR_STORE_DIR = "vector_store"
os.environ["GOOGLE_API_KEY"] = ""

# --- OCR fallback loader ---
def load_pdf_with_ocr(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    return text

# --- Document ingestion ---
def load_pdfs(folder):
    docs = []
    for file in os.listdir(folder):
        print(f"Processing {file}...")
        if file.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(folder, file))
                docs.extend(loader.load())
            except Exception:
                # Fallback to OCR if normal parsing fails
                text = load_pdf_with_ocr(os.path.join(folder, file))
                docs.append({"page_content": text, "metadata": {"source": file}})
    return docs

# --- Chunk & embed ---
def build_or_update_vectorstore():
    docs = load_pdfs(PDF_FOLDER)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists(VECTOR_STORE_DIR):
        db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
        db.add_documents(split_docs)
    else:
        db = Chroma.from_documents(split_docs, embeddings, persist_directory=VECTOR_STORE_DIR)

    return db

# --- Retrieval & QA ---
def query_rag(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
    retriever = db.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    qa = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa.invoke({"query": question})   

if __name__ == "__main__":
    # build_or_update_vectorstore()
    # print(query_rag("Summarize the main points of the documents."))
    print(query_rag("Which documents are related to use data privacy?"))
