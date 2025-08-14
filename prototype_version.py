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
from difflib import SequenceMatcher
from urllib.parse import urlparse, parse_qs

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
# 한국어 질의 전처리/확장
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

def generate_query_variants_korean(text: str, max_variants: int = 8) -> List[str]:
    base_raw = _basic_cleanup(text)
    stripped = re.sub(r"(관련|관한|관련된)\b", " ", base_raw)
    stripped = re.sub(r"(q\s*&\s*a|q\/a|q&a|qa|질문\s*답변|질답)", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    base = _normalize_korean_query(text)
    variants = [base]
    if stripped and stripped != base_raw:
        variants.append(_normalize_korean_query(stripped))
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
            href = href.lstrip("./")  # 안전하게 보정
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
            try:
                db.persist()
            except Exception:
                pass
        return db
    else:
        db = Chroma.from_documents(docs, EMBEDDINGS, persist_directory=dir_path)
        try:
            db.persist()
        except Exception:
            pass
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
# Scoring helpers
# =========================

def _to_similarity(dist: float, metric: str = "cosine") -> float:
    if metric == "cosine":
        return 1.0 - float(dist)  # cosine_distance = 1 - cosine_similarity
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
    st = (meta or {}).get("source_type", "")
    prior = 0.0
    if st in ("FAQS", "FAQ"):
        prior += 0.03
    elif st == "PDF":
        prior += 0.01
    return prior

def _fuse_title_body_scores(
    title_pairs: List[Tuple[Document, float]],
    body_map: Dict[str, float],
    w_title: float = 0.6,
    w_body: float = 0.4,
    use_rrf: bool = False,
) -> List[Tuple[Document, float]]:
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
# 🔧 동점 타이브레이커 + Lexical 게이트
# =========================
# PATCH-3: 부분 토큰(n-gram) 확장 지원

def _tokenize_simple(s: str) -> List[str]:
    """
    공백 기준 1차 토큰화 후, 각 토큰에 대해 2~3글자 n-gram 서브토큰을 추가한다.
    - 한국어/영문/숫자만 남기고 나머지는 제거
    - 너무 짧은 조각 폭증 방지 위해 길이 제한 및 dedup 적용
    """
    s = _basic_cleanup(s)
    base = [t for t in s.split() if t]
    cleaned = []
    for t in base:
        t = re.sub(r"[^0-9a-z가-힣]", "", t)  # 문자/숫자만
        if t:
            cleaned.append(t)

    out: set = set()
    for t in cleaned:
        out.add(t)
        L = len(t)
        # 2-gram
        if L >= 2:
            for i in range(L - 1):
                out.add(t[i:i+2])
        # 3-gram (길이 6 이상일 때만 추가로 생성해서 폭증 방지)
        if L >= 6:
            for i in range(L - 2):
                out.add(t[i:i+3])

    # 너무 짧은 단편은 제거(1글자 제외) — 한국어/영문 혼용 대비
    out = {tok for tok in out if len(tok) >= 2}
    return list(out)

def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    A, B = set(a), set(b)
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def _extract_recency_from_url(url: str) -> int:
    try:
        qs = parse_qs(urlparse(url).query)
        for key in ("nttId", "incdnSlno"):
            if key in qs and qs[key]:
                return int(qs[key][0])
    except Exception:
        pass
    try:
        qs = parse_qs(urlparse(url).query)
        pi = int(qs.get("pageIndex", [0])[0])
        return max(1, 100000 - pi)
    except Exception:
        return 0

def _source_type_weight(meta: Dict[str, Any]) -> float:
    st = (meta or {}).get("source_type", "")
    return {"FAQS": 3.0, "FAQ": 3.0, "DISPUTES": 2.0, "PDF": 1.0}.get(st, 0.0)

def _title_tiebreak_key(doc: Document, sim: float, query: str) -> tuple:
    title = (doc.metadata or {}).get("title") or ""
    url   = (doc.metadata or {}).get("source") or ""
    t_norm = _basic_cleanup(title)
    q_norm = _basic_cleanup(query)
    toks_t = _tokenize_simple(title)
    toks_q = _tokenize_simple(query)
    exact = 1 if t_norm == q_norm else 0
    starts = 1 if (t_norm.startswith(q_norm) or q_norm.startswith(t_norm)) else 0
    contains = 1 if (q_norm in t_norm or t_norm in q_norm) else 0
    jac = _jaccard(toks_t, toks_q)
    ratio = SequenceMatcher(None, q_norm, t_norm).ratio()
    st_w = _source_type_weight(doc.metadata)
    rec = _extract_recency_from_url(url)
    short_bonus = -len(t_norm)
    return (sim, exact, starts, contains, jac, ratio, st_w, rec, short_bonus)

# === 패치 1: 핵심 토큰 기반 불용어 제거 + 강화된 게이트 ===
STOPWORDS_KO = {
    "관련","관한","관련된","질문","답변","q&a","qa","q/a",
    "어떻게","되나요","되나","됩니까","무엇","무엇인가요",
    "가능","인가요","이다","하다","합니다","해요",
    "안내","유의사항","문의","바로가기"
}

def _content_tokens(s: str) -> List[str]:
    toks = _tokenize_simple(s)
    return [t for t in toks if t not in STOPWORDS_KO and len(t) >= 2]

def _passes_lexical_gate(title: str, query: str, min_jaccard: float = 0.08) -> bool:
    toks_t = _content_tokens(title)
    toks_q = _content_tokens(query)
    if not toks_t or not toks_q:
        return False
    overlap = len(set(toks_t) & set(toks_q))
    need = 1 if len(toks_q) <= 3 else 2  # 최소 겹침 개수
    if overlap < need:
        return False
    jac = _jaccard(toks_t, toks_q)
    t_norm = _basic_cleanup(title)
    q_norm = _basic_cleanup(query)
    contains = (q_norm in t_norm) or (t_norm in q_norm)
    return (jac >= min_jaccard) or contains

# =========================
# 문서 내부 로컬 관련도 (인덱스는 문서 단위 유지)
# =========================

STOP_PHRASES = {"고객센터","유의사항","문의","자료실","바로가기"}

def split_passages(text: str, target_len: int = 220, stride: int = 110) -> List[str]:
    """
    인덱스는 문서단위 유지. 런타임에만 문서 내부를 얕게 슬라이스.
    토큰은 _tokenize_simple 재사용(부분 n-gram 포함).
    """
    toks = _tokenize_simple(text)
    if not toks:
        return []
    passages = []
    for i in range(0, max(1, len(toks)-target_len+1), stride):
        chunk = " ".join(toks[i:i+target_len])
        if chunk:
            passages.append(chunk)
        if len(passages) >= 80:  # 안전 상한
            break
    if not passages:
        passages = [" ".join(toks[:target_len])]
    return passages

def keyword_overlap(q_tokens: List[str], p_tokens: List[str]) -> float:
    Q, P = set(q_tokens), set(p_tokens)
    inter = len(Q & P)
    return inter / max(1, len(Q))

def local_score_simple(query: str, passage: str) -> float:
    tq = _tokenize_simple(query)
    tp = _tokenize_simple(passage)
    if not tq or not tp:
        return 0.0
    jac = _jaccard(tq, tp)
    cover = keyword_overlap(tq, tp)
    return 0.5 * jac + 0.5 * cover

def boiler_penalty(text: str) -> float:
    t = _basic_cleanup(text)
    hits = sum(1 for s in STOP_PHRASES if s in t)
    return 0.02 * hits  # 최종 점수에서 뺄 예정

def doc_keyword_coverage(query: str, text: str) -> float:
    tq = set(_tokenize_simple(query))
    tt = set(_tokenize_simple(text))
    if not tq or not tt:
        return 0.0
    return len(tq & tt) / max(1, len(tq))

def rerank_inside_doc(doc: Document, query: str, emb_sim: float) -> Tuple[float, Dict[str, float]]:
    """
    최종 점수 = α*임베딩유사도 + β*문서내로컬최대 + γ*질의커버리지 + δ*제목보너스 - λ*보일러패널티
    """
    text = doc.page_content or ""
    passages = split_passages(text, target_len=220, stride=110)
    max_local = 0.0
    for p in passages:
        s = local_score_simple(query, p)
        if s > max_local:
            max_local = s

    cover = doc_keyword_coverage(query, text)
    title = (doc.metadata or {}).get("title", "") or ""
    title_hit = local_score_simple(query, title)
    penalty = boiler_penalty(text)

    # 하드 가드: 로컬·커버 모두 0이면 탈락
    if max_local == 0.0 and cover == 0.0:
        parts = {"emb":emb_sim, "local":0.0, "cover":0.0, "title":title_hit, "pen":penalty, "final":-1e9}
        return -1e9, parts

    # 가중치
    alpha, beta, gamma, delta, lamb = 0.6, 0.25, 0.10, 0.05, 1.0
    final = alpha*emb_sim + beta*max_local + gamma*cover + delta*title_hit - lamb*penalty
    parts = {"emb":emb_sim, "local":max_local, "cover":cover, "title":title_hit, "pen":penalty, "final":final}
    return final, parts

# =========================
# 2단계 검색 (제목 선필터 → 본문 재평가) + 한국어 쿼리 확장 + 적응형 k + 게이트
# =========================

def staged_retrieve(body_db: Chroma, title_db: Chroma, question: str, topk: int = 8,
                    title_k: int = 100, body_k_per_filter: int = 16,
                    metric: str = "cosine",
                    variant_agg: str = "softmax-mean",
                    use_rrf: bool = False) -> List[Document]:
    variants = generate_query_variants_korean(question)
    # print(f"[Query variants] {variants}")

    def run_title_search(k_val: int, use_gate: bool = True):
        title_best_sims: Dict[str, List[float]] = {}
        title_rep: Dict[str, Document] = {}
        for q in variants:
            res = title_db.similarity_search_with_score(q, k=k_val)
            for d, dist in res:
                if use_gate and not _passes_lexical_gate((d.metadata or {}).get("title",""), q):
                    continue
                src = (d.metadata or {}).get("source", "")
                sim = _to_similarity(dist, metric)
                if src not in title_best_sims:
                    title_best_sims[src] = []
                    title_rep[src] = d
                title_best_sims[src].append(sim)
        return title_best_sims, title_rep

    # 1차 시도 (gate ON)
    title_best_sims, title_rep = run_title_search(title_k, use_gate=True)

    # 실패 시 k 확장 (gate ON)
    if not title_best_sims:
        for k_try in (300, 800):
            title_best_sims, title_rep = run_title_search(k_try, use_gate=True)
            if title_best_sims:
                print(f"[title-stage] expanded k → {k_try}")
                break

    # 그래도 실패면 gate OFF fallback
    if not title_best_sims:
        title_best_sims, title_rep = run_title_search(800, use_gate=False)
        if title_best_sims:
            print("[title-stage] no gate fallback used")

    if not title_best_sims:
        return []

    title_final: Dict[str, float] = {src: _aggregate_variants_to_best(sims, mode=variant_agg)
                                     for src, sims in title_best_sims.items()}

    # Stage 2: 본문 재검색 (제목 후보만)
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

    # 결합 + 타이브레이크(기존)
    title_pairs = [(title_rep[src], title_final.get(src, 0.0)) for src in candidate_sources]
    merged = _fuse_title_body_scores(title_pairs, body_final, w_title=0.6, w_body=0.4, use_rrf=use_rrf)
    merged.sort(key=lambda x: _title_tiebreak_key(x[0], x[1], question), reverse=True)

    # 문서 내부 로컬 관련도 기반 재랭크(인덱스는 문서 단위 유지)
    per_source_best: Dict[str, Tuple[Document, float, Dict[str, float]]] = {}

    for doc, _ in merged:
        src = (doc.metadata or {}).get("source", "")
        if src in per_source_best:
            continue  # source 당 하나만 평가(비용 절감)

        # 쿼리 기준으로 그 source 내에서 가장 관련있는 본문 문서를 1개 고름
        try:
            q_best = body_db.similarity_search_with_score(
                question, k=3, filter={"source": src}
            )
        except Exception:
            q_best = []
        if not q_best:
            # fallback: 제목 대표의 텍스트 기준으로 탐색
            try:
                q_best = body_db.similarity_search_with_score(doc.page_content, k=1, filter={"source": src})
            except Exception:
                q_best = []

        if not q_best:
            continue

        bd, dist = q_best[0]
        emb_sim = _to_similarity(dist, metric)
        final_score, parts = rerank_inside_doc(bd, question, emb_sim)
        per_source_best[src] = (bd, final_score, parts)

    # 최종 상위 topk 선택
    ranked = sorted(per_source_best.values(), key=lambda x: x[1], reverse=True)
    selected_docs: List[Document] = []
    for bd, score, parts in ranked[:topk]:
        # 디버깅 확인용
        # print(f"[final-rerank] src={ (bd.metadata or {}).get('source','') } "
        #       f"score={score:.4f} parts={parts}")
        selected_docs.append(bd)

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
# 벡터스토어/제목 DB 점검 & 디버그
# =========================

def assert_vectorstores_ready():
    body = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=EMBEDDINGS)
    title = Chroma(persist_directory=TITLE_VECTOR_STORE_DIR, embedding_function=EMBEDDINGS)
    try:
        body_count = len(body.get(include=[] )["ids"])
    except Exception:
        body_count = -1
    try:
        title_count = len(title.get(include=[] )["ids"])
    except Exception:
        title_count = -1
    print(f"[VS READY] body={body_count}, title={title_count}")
    if body_count <= 0 or title_count <= 0:
        print("-> 벡터스토어가 비어있습니다. build_or_update_vectorstore(force_refresh=True)를 먼저 호출하세요.")

def debug_find_by_title(substr: str, k: int = 10):
    title_db = Chroma(persist_directory=TITLE_VECTOR_STORE_DIR, embedding_function=EMBEDDINGS)
    q = _basic_cleanup(substr)

    def run(kv: int, use_gate: bool = True):
        hits = title_db.similarity_search_with_score(q, k=kv)
        if not hits:
            return []
        if use_gate:
            hits = [(d, dist) for d, dist in hits if _passes_lexical_gate((d.metadata or {}).get("title",""), q)]
        scored = [(d, 1 - float(dist)) for d, dist in hits]  # cosine 가정
        scored.sort(key=lambda x: _title_tiebreak_key(x[0], x[1], substr), reverse=True)
        return scored

    print(f'--- search titles for: "{substr}" ---')
    scored = run(max(k, 100), use_gate=True)
    if not scored:
        print("(gate pass=0) retry with larger k and relaxed gate…")
        scored = run(800, use_gate=False)
    if not scored:
        print("(no hits after fallback)")
        return
    for d, sim in scored[:k]:
        print(f"{sim:.3f} | {d.metadata.get('title')} | {d.metadata.get('source')}")

def _count_chroma(persist_dir: str) -> int:
    try:
        db = Chroma(persist_directory=persist_dir, embedding_function=EMBEDDINGS)
        return len(db.get(include=[])["ids"])
    except Exception:
        return 0

# =========================
# QA (RAG) – 커스텀 2단계 검색 + 커스텀 프롬프트
# =========================

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

def query_rag(question: str):
    body_db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=EMBEDDINGS)
    title_db = Chroma(persist_directory=TITLE_VECTOR_STORE_DIR, embedding_function=EMBEDDINGS)

    top_docs = staged_retrieve(
        body_db, title_db, question,
        topk=8, title_k=100, body_k_per_filter=16,
        metric="cosine", variant_agg="softmax-mean", use_rrf=False
    )
    if not top_docs:
        return {"answer": "", "sources": []}

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

    stuff_chain = create_stuff_documents_chain(LLM, prompt)
    answer_text = stuff_chain.invoke({"question": question, "context": top_docs})

    # 깔끔한 출처 목록 구성
    sources = []
    seen = set()
    for d in top_docs:
        src = (d.metadata or {}).get("source", "")
        title = (d.metadata or {}).get("title") or src or "untitled"
        if src and src not in seen:
            seen.add(src)
            sources.append({"title": title, "url": src})

    return {"answer": answer_text, "sources": sources}

# 무응답/불확실 답변 패턴 감지
_NOANSWER_PATTERNS = (
    "찾지 못했습니다", "확인되지 않았습니다", "모르겠습니다",
    "근거가 없습니다", "제공된 문서 컨텍스트에서", "충분한 정보가", "찾을 수 없습니다."
)

def _is_noanswer(text: str) -> bool:
    if not text or not text.strip():
        return True
    t = text.strip()
    return any(pat in t for pat in _NOANSWER_PATTERNS)

# =========================
# Main
# =========================

if __name__ == "__main__":
    # 제목 벡터스토어가 비어 있으면 자동 복구 (캐시 기반)
    title_count_before = _count_chroma(TITLE_VECTOR_STORE_DIR)
    if title_count_before == 0:
        print("[INIT] Title vector store is empty → building title/body stores from caches…")
        build_or_update_vectorstore(force_refresh=False)

    print("\n" + "="*68)
    print(" 금융 분쟁/FAQ RAG 어시스턴트 ".center(68, " "))
    print("="*68)
    print("질문을 입력하세요. (종료: exit / quit / q / 종료)")
    print("-" * 68)

    while True:
        try:
            q = input("\n질문 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n종료합니다.")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit", "q", "종료"):
            print("종료합니다.")
            break

        t0 = time.time()
        result = query_rag(q)
        dt = time.time() - t0

        answer = (result or {}).get("answer", "").strip()
        sources = (result or {}).get("sources", [])

        print("\n" + "="*68)
        print("답변".center(68, " "))
        print("-"*68)
        if _is_noanswer(answer):
            print("죄송합니다. 제공된 문서 컨텍스트에서 답을 찾지 못했습니다.")
            show_sources = False
        else:
            print(answer)
            show_sources = True

        print("\n" + "출처".center(68, " "))
        print("-"*68)
        if show_sources and sources:
            for i, s in enumerate(sources, 1):
                title = s.get("title", "untitled")
                url = s.get("url", "")
                print(f"{i}. {title}\n   ↳ {url}")
        else:
            print("표시할 출처가 없습니다.")

        print("\n처리시간: {:.2f}s".format(dt))
        print("="*68)
