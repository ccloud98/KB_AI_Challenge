import os
import pickle
import time
import re
import unicodedata
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
# 문서 단위 유지 (분할 X)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple

# =========================
# Config & Globals
# =========================

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

VECTOR_STORE_DIR = "vector_store"                 # 본문(문서) 벡터
TITLE_VECTOR_STORE_DIR = "vector_store_titles"    # 제목 전용 벡터

WEB_CACHE_FILE = "web_cache.pkl"
FAQS_CACHE_FILE = "faqs_cache.pkl"

load_dotenv()
_api_key = os.getenv("GOOGLE_API_KEY")
if not isinstance(_api_key, str) or not _api_key.strip():
    raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다. .env 또는 환경변수에 유효한 키를 넣어주세요.")
os.environ["GOOGLE_API_KEY"] = _api_key.strip()

# 공용 임베딩/LLM (재사용)
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# =========================
# 한국어 질의 전처리 (가벼운 규칙 기반)
# =========================

_JOSA = r"(은|는|이|가|을|를|에|에서|으로|와|과|도|만|부터|까지|마다|처럼|보다|의|로서|이라면|이라도|이며|이고)$"
_EOMI = r"(하다|합니다|해요|했나요|되나요|되었나요|됩니까|인가요|인가요\?|인가|일까요|일까|였다|였다가|되었다|한다)$"

def _strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def _basic_cleanup(text: str) -> str:
    text = _strip_accents(text).lower()
    text = text.replace("％", "%").replace("•", " ").replace("·", " ")
    text = re.sub(r"[“”\"'`]", " ", text)
    text = re.sub(r"[()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _light_morph_simplify_token(tok: str) -> str:
    tok = re.sub(_JOSA, "", tok)
    tok = re.sub(_EOMI, "하다", tok)
    return tok

def _normalize_korean_query(text: str) -> str:
    t = _basic_cleanup(text)
    t = re.sub(r"(\d+)\s*퍼센트", r"\1%", t)
    t = re.sub(r"(\d+)\s*퍼\s*센\s*트", r"\1%", t)
    t = t.replace("%p", "%")
    tokens = []
    for tok in t.split(" "):
        if not tok:
            continue
        if re.search(r"[가-힣]", tok):
            tok = _light_morph_simplify_token(tok)
        tokens.append(tok)
    t = " ".join(tokens)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def generate_query_variants_korean(text: str, max_variants: int = 4) -> List[str]:
    base = _normalize_korean_query(text)
    variants = [base]
    compact = re.sub(r"\s+", "", base)
    if compact != base:
        variants.append(compact)
    no_punct = re.sub(r"[-/]", " ", base)
    no_punct = re.sub(r"\s+", " ", no_punct).strip()
    if no_punct not in variants:
        variants.append(no_punct)
    attach_num_unit = re.sub(r"(\d+)\s*%", r"\1%", base)
    if attach_num_unit not in variants:
        variants.append(attach_num_unit)
    uniq = []
    for v in variants:
        if v and v not in uniq:
            uniq.append(v)
    return uniq[:max_variants]

# =========================
# Web scraping
# =========================

def get_table_links(page_index, menu_no, base_list_url, base_url):
    params = {"menuNo": menu_no, "pageIndex": page_index}
    resp = requests.get(base_list_url, params=params, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a_tag in soup.select("a[href*='view.do']"):
        href = a_tag.get("href")
        if not href:
            continue
        if href.startswith("."):
            href = href[1:]
        links.append(base_url + href)
    return links

def get_web_page_details(url, source_type):
    resp = requests.get(url, timeout=15)
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

    all_docs: List[Document] = []
    for page in range(1, web_source_confing.num_pages + 1):
        print(f"Fetching list page {page}...")
        case_links = get_table_links(
            page,
            web_source_confing.menu_no,
            web_source_confing.base_list_url,
            web_source_confing.base_url,
        )
        for link in case_links:
            if link in processed_urls:
                continue
            print(f"  Fetching case: {link}")
            try:
                doc = get_web_page_details(link, web_source_confing.source_type)
                all_docs.append(doc)
                processed_urls.add(link)
                time.sleep(0.2)
            except Exception as e:
                print(f"    Failed to fetch {link}: {e}")

    with open(web_source_confing.cache_file, "wb") as f:
        pickle.dump(all_docs, f)
    with open(PROCESSED_URLS_FILE, "wb") as f:
        pickle.dump(processed_urls, f)
    print(f"Saved {len(all_docs)} {web_source_confing.source_type} docs to cache.")
    return all_docs

# =========================
# PDF loading with caching and OCR fallback
# =========================

def load_pdf_with_ocr(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img, lang="kor+eng")
    return text

def load_pdfs(folder, force_refresh=False):
    if not force_refresh and os.path.exists(PDF_CACHE_FILE):
        print("Loading PDF docs from cache...")
        with open(PDF_CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Loading PDFs from folder...")
    docs: List[Document] = []
    if not os.path.isdir(folder):
        print(f"  PDF 폴더가 없습니다: {folder}")
        with open(PDF_CACHE_FILE, "wb") as f:
            pickle.dump(docs, f)
        return docs

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
            try:
                text = load_pdf_with_ocr(file_path)
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": os.path.join(folder, file), "source_type": "PDF_OCR"},
                    )
                )
            except Exception as e2:
                print(f"  OCR 실패: {file} ({e2})")
    with open(PDF_CACHE_FILE, "wb") as f:
        pickle.dump(docs, f)
    print(f"Saved {len(docs)} PDF-page docs to cache.")
    return docs

def combine_pdf_pages_by_file(page_docs: List[Document]) -> List[Document]:
    by_file: Dict[str, List[str]] = {}
    for d in page_docs:
        src = d.metadata.get("source", "unknown.pdf")
        by_file.setdefault(src, []).append(d.page_content)

    combined: List[Document] = []
    for src, pages in by_file.items():
        full_text = "\n\n".join(pages)
        source_type = "PDF_OCR" if any("PDF_OCR" == d.metadata.get("source_type") for d in page_docs if d.metadata.get("source") == src) else "PDF"
        combined.append(
            Document(
                page_content=full_text,
                metadata={"source": src, "title": os.path.basename(src), "source_type": source_type},
            )
        )
    print(f"Combined into {len(combined)} PDF-file docs.")
    return combined

# =========================
# Build or update vector store (문서 단위 + 제목 인덱스)
# =========================

def _ensure_chroma(dir_path: str, docs: List[Document]) -> Chroma:
    if os.path.exists(dir_path):
        db = Chroma(persist_directory=dir_path, embedding_function=EMBEDDINGS)
        if docs:
            db.add_documents(docs)
            try: db.persist()
            except Exception: pass
        return db
    else:
        db = Chroma.from_documents(docs, EMBEDDINGS, persist_directory=dir_path)
        try: db.persist()
        except Exception: pass
        return db

def build_or_update_vectorstore(force_refresh: bool = False):
    pdf_page_docs = load_pdfs(PDF_FOLDER, force_refresh=force_refresh)
    pdf_docs = combine_pdf_pages_by_file(pdf_page_docs)

    disputes_config = WebSourceConfig(
        cache_file=WEB_CACHE_FILE,
        base_url="https://www.fss.or.kr/fss/job/fvsttPrcdnt",
        base_list_url="https://www.fss.or.kr/fss/job/fvsttPrcdnt/list.do",
        source_type="DISPUTES",
        num_pages=56,
        menu_no=200179,
    )
    faqs_config = WebSourceConfig(
        cache_file=FAQS_CACHE_FILE,
        base_url="https://www.fss.or.kr",
        base_list_url="https://www.fss.or.kr/fss/bbs/B0000172/list.do",
        source_type="FAQS",
        num_pages=94,
        menu_no=200202,
    )
    dispute_docs = scrape_web_docs(disputes_config, force_refresh=force_refresh)
    faq_docs = scrape_web_docs(faqs_config, force_refresh=force_refresh)

    all_docs = pdf_docs + dispute_docs + faq_docs

    title_docs = []
    for d in all_docs:
        title = (d.metadata or {}).get("title") or (d.metadata or {}).get("source") or ""
        title_norm = _basic_cleanup(title)
        title_docs.append(Document(page_content=title_norm, metadata=d.metadata))

    body_db = _ensure_chroma(VECTOR_STORE_DIR, all_docs)
    title_db = _ensure_chroma(TITLE_VECTOR_STORE_DIR, title_docs)
    return body_db, title_db

# =========================
# Scoring helpers (정확도 강화)
# =========================

def _to_similarity(dist: float, metric: str = "cosine") -> float:
    """distance -> similarity 변환. 기본 cosine 가정."""
    if metric == "cosine":
        return 1.0 - float(dist)           # cosine_distance = 1 - cosine_similarity
    elif metric in ("l2", "euclidean"):
        return 1.0 / (1.0 + float(dist))
    else:
        return 1.0 / (1.0 + float(dist))

def _zscore(xs: List[float]) -> List[float]:
    if not xs:
        return xs
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(1, len(xs)-1)
    sd = v ** 0.5
    if sd < 1e-12:
        return [0.0 for _ in xs]
    return [(x - m) / sd for x in xs]

def _softmax(xs: List[float], tau: float = 1.0) -> List[float]:
    import math
    if not xs:
        return xs
    mx = max(xs)
    exps = [math.exp((x - mx) / max(1e-8, tau)) for x in xs]
    s = sum(exps)
    return [e / max(1e-12, s) for e in exps]

def _aggregate_variants_to_best(sim_list: List[float], mode: str = "softmax-mean") -> float:
    if not sim_list:
        return 0.0
    if mode == "max":
        return max(sim_list)
    if mode == "mean":
        return sum(sim_list) / len(sim_list)
    if mode == "softmax-mean":
        w = _softmax(sim_list, tau=0.25)
        return sum(si * wi for si, wi in zip(sim_list, w))
    return max(sim_list)

def _apply_priors(meta: Dict[str, Any]) -> float:
    """source_type/최신성 등 미세 가산. 값은 과도하지 않게."""
    st = (meta or {}).get("source_type", "")
    prior = 0.0
    if st in ("FAQS", "FAQ"):
        prior += 0.03
    elif st == "PDF":
        prior += 0.01
    return prior

def _fuse_title_body_scores(
    title_pairs: List[Tuple[Document, float]],   # (doc, title_sim)
    body_map: Dict[str, float],                  # source -> body_sim
    w_title: float = 0.6,
    w_body: float = 0.4,
    use_rrf: bool = False,
) -> List[Tuple[Document, float]]:
    """z-score 가중합 또는 RRF로 결합"""
    titles = [s for (_, s) in title_pairs]
    bodies = [body_map.get((d.metadata or {}).get("source", ""), 0.0) for (d, _) in title_pairs]

    if use_rrf:
        def _ranks(vals):
            order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
            rank = [0]*len(vals)
            for r, i in enumerate(order):
                rank[i] = r + 1
            return rank
        k = 60
        rt = _ranks(titles)
        rb = _ranks(bodies)
        fused = []
        for (doc, _), rti, rbi in zip(title_pairs, rt, rb):
            score = (1.0/(k + rti)) + (1.0/(k + rbi)) + _apply_priors(doc.metadata)
            fused.append((doc, score))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    zt = _zscore(titles)
    zb = _zscore(bodies)
    fused = []
    for (doc, _), ts, bs in zip(title_pairs, zt, zb):
        score = w_title*ts + w_body*bs + _apply_priors(doc.metadata)
        fused.append((doc, score))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused

# =========================
# 2단계 검색 (제목 선필터 → 본문 재평가) + 한국어 쿼리 확장 + 고급 스코어링
# =========================

def staged_retrieve(body_db: Chroma, title_db: Chroma, question: str, topk: int = 8,
                    title_k: int = 100, body_k_per_filter: int = 16,
                    metric: str = "cosine",
                    variant_agg: str = "softmax-mean",
                    use_rrf: bool = False) -> List[Document]:
    variants = generate_query_variants_korean(question)
    print(f"[Query variants] {variants}")

    # --- Stage 1: 제목 검색 (변형 통합; source별 best title_sim) ---
    title_best_sims: Dict[str, List[float]] = {}  # source -> [sim per variant]
    title_rep: Dict[str, Document] = {}
    for q in variants:
        res = title_db.similarity_search_with_score(q, k=title_k)
        for d, dist in res:
            src = (d.metadata or {}).get("source", "")
            sim = _to_similarity(dist, metric)
            if src not in title_best_sims:
                title_best_sims[src] = []
                title_rep[src] = d
            title_best_sims[src].append(sim)

    if not title_best_sims:
        return []

    # 변형 집계
    title_final: Dict[str, float] = {src: _aggregate_variants_to_best(sims, mode=variant_agg)
                                     for src, sims in title_best_sims.items()}

    # --- Stage 2: 본문 재검색 (제목 후보만; 변형 통합) ---
    candidate_sources = list(title_final.keys())
    body_sims: Dict[str, List[float]] = {src: [] for src in candidate_sources}
    for q in variants:
        try:
            res = body_db.similarity_search_with_score(
                q, k=body_k_per_filter, filter={"source": {"$in": candidate_sources}}
            )
        except Exception:
            res = body_db.similarity_search_with_score(q, k=max(body_k_per_filter, 64))
            res = [(d, dist) for d, dist in res if (d.metadata or {}).get("source", "") in candidate_sources]
        for d, dist in res:
            src = (d.metadata or {}).get("source", "")
            body_sims[src].append(_to_similarity(dist, metric))

    body_final: Dict[str, float] = {src: _aggregate_variants_to_best(sims, mode=variant_agg)
                                    for src, sims in body_sims.items()}

    # --- 결합 (z-score 가중합 or RRF) ---
    title_pairs = [(title_rep[src], title_final.get(src, 0.0)) for src in candidate_sources]
    merged = _fuse_title_body_scores(title_pairs, body_final, w_title=0.6, w_body=0.4, use_rrf=use_rrf)

    # --- 최종 상위 topk의 '본문 문서' 반환 ---
    selected_docs = []
    used_src = set()
    for doc, _ in merged:
        src = (doc.metadata or {}).get("source", "")
        if src in used_src:
            continue
        try:
            body_res = body_db.similarity_search(doc.page_content, k=1, filter={"source": src})
        except Exception:
            body_res = []
        if not body_res:
            tmp = body_db.similarity_search(doc.page_content, k=8)
            tmp = [d for d in tmp if (d.metadata or {}).get("source", "") == src]
            if tmp:
                body_res = [tmp[0]]
        if body_res:
            selected_docs.append(body_res[0])
            used_src.add(src)
        if len(selected_docs) >= topk:
            break
    return selected_docs

# =========================
# Pair-by-pair cache viewer (2개씩 보기)
# =========================

def _print_doc_block(doc: Document, idx: int, full: bool = False, max_chars: int = 1200) -> None:
    meta = doc.metadata or {}
    title = meta.get("title") or meta.get("source") or "(no title)"
    stype = meta.get("source_type", "")
    text = doc.page_content or ""
    if not full and len(text) > max_chars:
        text = text[:max_chars] + "…"
    print(f"[{idx}] title/source: {title}")
    print(f"     type: {stype}")
    print("----- TEXT BEGIN -----")
    print(text)
    print("------ TEXT END ------\n")

def _load_pickle(pkl_path: str) -> Any:
    if not os.path.exists(pkl_path):
        print(f"(없음) {pkl_path}")
        return None
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"{pkl_path} 로드 오류: {e}")
        return None

def view_cache_pairs(pkl_path: str, start: int = 0, full: bool = False, max_chars: int = 1200) -> None:
    data = _load_pickle(pkl_path)
    if data is None:
        return
    if isinstance(data, set):
        data = list(data)
    total = len(data) if hasattr(data, "__len__") else None
    if total is None:
        print(f"지원하지 않는 타입: {type(data)}")
        print(data)
        return
    if start >= total:
        print(f"start({start})가 총 개수({total}) 이상입니다.")
        return
    end = min(start + 2, total)
    print(f"\n=== {pkl_path} [{start} ~ {end - 1}] / 총 {total}개 ===")
    if total > 0 and isinstance(data[0], Document):
        for i in range(start, end):
            _print_doc_block(data[i], i, full=full, max_chars=max_chars)
    else:
        for i in range(start, end):
            print(f"[{i}] {data[i]}")
        print()

def view_pdf_cache_pairs(start: int = 0, full: bool = False, max_chars: int = 1200) -> None:
    view_cache_pairs(PDF_CACHE_FILE, start=start, full=full, max_chars=max_chars)

def view_web_cache_pairs(start: int = 0, full: bool = False, max_chars: int = 1200) -> None:
    view_cache_pairs("web_cache.pkl", start=start, full=full, max_chars=max_chars)

def view_faqs_cache_pairs(start: int = 0, full: bool = False, max_chars: int = 1200) -> None:
    view_cache_pairs("faqs_cache.pkl", start=start, full=full, max_chars=max_chars)

def view_processed_urls_pairs(start: int = 0, full: bool = False, max_chars: int = 1200) -> None:
    view_cache_pairs(PROCESSED_URLS_FILE, start=start, full=full, max_chars=max_chars)

# =========================
# QA (RAG) – 커스텀 2단계 검색 + 커스텀 프롬프트
# =========================

def query_rag(question: str):
    body_db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=EMBEDDINGS)
    title_db = Chroma(persist_directory=TITLE_VECTOR_STORE_DIR, embedding_function=EMBEDDINGS)

    top_docs = staged_retrieve(
        body_db, title_db, question,
        topk=8, title_k=100, body_k_per_filter=16,
        metric="cosine", variant_agg="softmax-mean", use_rrf=False
    )
    if not top_docs:
        print("검색 결과가 없습니다.")
        return ""

    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain.chains.question_answering import load_qa_chain

    system_text = (
        "당신은 금융 분쟁/FAQ 문서를 근거로 답하는 한국어 어시스턴트입니다. "
        "다음 규칙을 지키세요:\n"
        "1) 아래 제공되는 CONTEXT만 근거로 답변할 것\n"
        "2) 한국어 용어를 일관되게 사용할 것\n"
        "3) 모르면 모른다고 말할 것\n"
        "4) 마지막에 참고한 출처 제목을 bullet로 나열할 것"
    )
    human_text = (
        "질문:\n{question}\n\n"
        "CONTEXT(발췌):\n{context}\n\n"
        "위 CONTEXT만을 근거로 한국어로 간결하고 정확하게 답하세요."
    )
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_text),
        HumanMessagePromptTemplate.from_template(human_text),
    ])

    chain = load_qa_chain(LLM, chain_type="stuff", chain_type_kwargs={"prompt": prompt})

    context_text = "\n\n---\n\n".join([d.page_content for d in top_docs])
    result = chain.invoke({"input_documents": [], "question": question, "context": context_text})

    answer = result["output_text"]
    print("Answer:", answer)
    print("\nSource docs:")
    for d in top_docs:
        print("-", d.metadata.get("title") or d.metadata.get("source", "unknown"))
    return answer

# =========================
# Main
# =========================

if __name__ == "__main__":
    # 1) 최초/갱신 빌드: 문서 단위 색인 + 제목 인덱스 동시 구축
    # body_db, title_db = build_or_update_vectorstore(force_refresh=True)

    # PDF 캐시 0~1번 (2개) 미리보기
    view_pdf_cache_pairs(start=0, full=False)

    # PDF 캐시 2~3번 (다음 2개) 전체 출력
    view_pdf_cache_pairs(start=2, full=True)

    # 분쟁사례 웹 캐시 0~1번
    view_web_cache_pairs(start=0, full=False)

    # FAQ 캐시 10~11번
    view_faqs_cache_pairs(start=10, full=False)

    # 처리된 URL 0~1번
    view_processed_urls_pairs(start=0)

    # 2) 샘플 질의
    result = query_rag("카드단말기 IC전환")
    print(result)
