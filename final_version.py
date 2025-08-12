import os
import pickle
import time
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from dataclasses import dataclass
from dotenv import load_dotenv
import os


@dataclass
class WebSourceConfig:
    cache_file: str
    base_url: str
    base_list_url: str
    source_type: str
    num_pages: int
    menu_no: int


PDF_FOLDER = "KB_web_docs"
PDF_CACHE_FILE = "pdf_cache.pkl"

PROCESSED_URLS_FILE = "processed_urls.pkl"

VECTOR_STORE_DIR = "vector_store"

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


def get_table_links(page_index, menu_no, base_list_url, base_url):
    params = {"menuNo": menu_no, "pageIndex": page_index}
    resp = requests.get(base_list_url, params=params)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a_tag in soup.select("table tr td a"):
        href = a_tag.get("href")
        if href and "view.do" in href:
            if href[0] == ".":
                href = href[1:]
            links.append(base_url + href)
    return links


def get_web_page_details(url, source_type):
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    subject_tag = soup.find("h2", class_="subject")
    title = subject_tag.get_text(strip=True) if subject_tag else "No subject found"

    content_tag = soup.find("div", class_="dbdata")
    content_text = (
        content_tag.get_text(separator="\n", strip=True)
        if content_tag
        else "No content found"
    )

    print(f"Fetched case: {title}")
    print(content_text[:100])

    return Document(
        page_content=content_text,
        metadata={"source": url, "title": title, "source_type": source_type},
    )


def scrape_web_docs(web_source_confing: WebSourceConfig, force_refresh=False):
    if not force_refresh and os.path.exists(web_source_confing.cache_file):
        print(f"Loading {web_source_confing.source_type} docs from cache...")
        with open(web_source_confing.cache_file, "rb") as f:
            return pickle.load(f)

    print(f"Scraping {web_source_confing.source_type} docs from website...")
    if os.path.exists(PROCESSED_URLS_FILE):
        with open(PROCESSED_URLS_FILE, "rb") as f:
            processed_urls = pickle.load(f)
    else:
        processed_urls = set()

    all_docs = []
    for page in range(1, web_source_confing.num_pages + 1):
        print(f"Fetching list page {page}...")
        case_links = get_table_links(
            page,
            web_source_confing.menu_no,
            web_source_confing.base_list_url,
            web_source_confing.base_url,
        )

        processed_urls.update(case_links)

        for link in case_links:
            print(f"  Fetching case: {link}")
            try:
                doc = get_web_page_details(link, web_source_confing.source_type)
                all_docs.append(doc)
                time.sleep(0.5)
            except Exception as e:
                print(f"    Failed to fetch {link}: {e}")

    # Save to cache
    with open(web_source_confing.cache_file, "wb") as f:
        pickle.dump(all_docs, f)
    with open(PROCESSED_URLS_FILE, "wb") as f:
        pickle.dump(processed_urls, f)
    print(f"Saved {len(all_docs)} {web_source_confing.source_type} docs to cache.")
    return all_docs


# --- PDF loading with caching and OCR fallback ---


def load_pdf_with_ocr(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    return text


def load_pdfs(folder, force_refresh=False):
    if not force_refresh and os.path.exists(PDF_CACHE_FILE):
        print("Loading PDF docs from cache...")
        with open(PDF_CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Loading PDFs from folder...")
    docs = []
    for file in os.listdir(folder):
        if not file.lower().endswith(".pdf"):
            continue
        print(f"Processing {file}...")
        file_path = os.path.join(folder, file)
        try:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        except Exception:
            print(f"  Fallback to OCR for {file}")
            text = load_pdf_with_ocr(file_path)
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": file, "source_type": "PDF_OCR"},
                )
            )
    # Save to cache
    with open(PDF_CACHE_FILE, "wb") as f:
        pickle.dump(docs, f)
    print(f"Saved {len(docs)} PDF docs to cache.")
    return docs


# --- Build or update vector store ---


def build_or_update_vectorstore(force_refresh=False):
    pdf_docs = load_pdfs(PDF_FOLDER, force_refresh=force_refresh)
    disputes_config = WebSourceConfig(
        cache_file="web_cache.pkl",
        base_url="https://www.fss.or.kr/fss/job/fvsttPrcdnt",
        base_list_url="https://www.fss.or.kr/fss/job/fvsttPrcdnt/list.do",
        source_type="DISPUTES",
        num_pages=56,
        menu_no=200179,
    )
    faqs_config = WebSourceConfig(
        cache_file="faqs_cache.pkl",
        base_url="https://www.fss.or.kr",
        base_list_url="https://www.fss.or.kr/fss/bbs/B0000172/list.do",
        source_type="FAQS",
        num_pages=94,
        menu_no=200202,
    )

    dispute_docs = scrape_web_docs(disputes_config, force_refresh=force_refresh)
    faq_docs = scrape_web_docs(faqs_config, force_refresh=force_refresh)

    all_docs = pdf_docs + dispute_docs + faq_docs

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists(VECTOR_STORE_DIR):
        print("Loading existing vector store and adding documents...")
        db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
        db.add_documents(split_docs)
    else:
        print("Creating new vector store...")
        db = Chroma.from_documents(
            split_docs, embeddings, persist_directory=VECTOR_STORE_DIR
        )

    return db


# --- Query RAG ---
def query_rag(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    from langchain.chains import RetrievalQA

    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    result = qa.invoke({"query": question})
    print("Answer:", result["result"])
    print("\nSource docs:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "unknown"))
    return result["result"]


if __name__ == "__main__":

    # Uncomment to build a new vector store or update the existing one

    # build_or_update_vectorstore(force_refresh=False)

    # Sample query
    result = query_rag("각 서비스별 개인정보 처리 방침의 차이점은 무엇인가요??")
    print(result)
