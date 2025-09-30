#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
json_rag_win.py (v7 - FAISS instead of ChromaDB)
- Replaces ChromaDB with FAISS for better consistency across Windows systems
- Keeps all v6 features: soft filters, signal extraction, LLM integration
- Same CLI interface and functionality
"""

import os, sys, json, argparse, re, multiprocessing
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import hashlib
import gc
import threading
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ---------------- CPU / batching config ----------------
CPU_COUNT = multiprocessing.cpu_count()
OPTIMAL_THREADS = max(1, CPU_COUNT - 1)
BATCH_SIZE = min(1024, max(50, CPU_COUNT * 10))
CHUNK_SIZE = 2000

LIST_TOTAL_Q = re.compile(
    r"\b(list|show|names?|total|how\s+many|count|number\s+of)\b.*\bcompan(y|ies)\b",
    re.IGNORECASE,
)

SYNONYMS = {
    "rd": ["research and development", "R&D", "rd facility", "rd_nabl_accredited", "high voltage lab", "laboratory"],
    "testing": ["testing facility", "test lab", "nabl", "testing capabilities", "electrical testing", "insulation testing"],
    "certification": ["ISO", "certifications", "certificate_number", "certification_type_master"],
    "product": ["products", "services", "product_name", "salient_features", "transformer", "high voltage transformer"],
}
LOCATION_KEYS = ["city", "district", "state", "country", "pincode"]

# ---------------- Embedding model cache ----------------
_embedding_model = None
_embedding_lock = threading.Lock()

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                cache_dir = Path.home() / ".cache" / "sentence_transformers"
                cache_dir.mkdir(parents=True, exist_ok=True)
                m = SentenceTransformer(
                    'sentence-transformers/all-MiniLM-L6-v2', 
                    device='cpu',
                    cache_folder=str(cache_dir)
                )
                m.eval()
                _embedding_model = m
    return _embedding_model

# ---------------- FAISS Store ----------------
class FAISSStore:
    """FAISS-backed vector store with metadata support"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine similarity after normalization
        self.metadatas = []
        self.documents = []
        self.ids = []
    
    def add(self, ids: List[str], documents: List[str], 
            embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """Add documents to index"""
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
    
    def query(self, query_embedding: np.ndarray, n_results: int = 10) -> Dict[str, List]:
        """Search - ChromaDB-compatible interface"""
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, min(n_results, len(self.ids)))
        
        # ChromaDB format
        return {
            "ids": [[self.ids[i] for i in indices[0] if i < len(self.ids)]],
            "documents": [[self.documents[i] for i in indices[0] if i < len(self.documents)]],
            "metadatas": [[self.metadatas[i] for i in indices[0] if i < len(self.metadatas)]],
            "distances": distances.tolist()
        }
    
    def count(self) -> int:
        return self.index.ntotal
    
    def get(self, include: List[str] = None) -> Dict[str, List]:
        """Get all documents - ChromaDB-compatible"""
        result = {"ids": self.ids}
        if include and "metadatas" in include:
            result["metadatas"] = self.metadatas
        if include and "documents" in include:
            result["documents"] = self.documents
        return result
    
    def save(self, path: Path):
        """Save index and metadata"""
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump({
                "ids": self.ids,
                "documents": self.documents,
                "metadatas": self.metadatas,
                "dimension": self.dimension
            }, f)
        print(f"[FAISS] Saved {self.count()} documents to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'FAISSStore':
        """Load index and metadata"""
        index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.pkl", "rb") as f:
            data = pickle.load(f)
        
        store = cls(dimension=data["dimension"])
        store.index = index
        store.ids = data["ids"]
        store.documents = data["documents"]
        store.metadatas = data["metadatas"]
        print(f"[FAISS] Loaded {store.count()} documents from {path}")
        return store

# ---------------- Utils ----------------
ISO_PATTERN = re.compile(r"(iso\s*\d{3,5})", re.IGNORECASE)
LOC_IN_PATTERN = re.compile(r"\bin\s+([a-zA-Z][\w\- ]{2,})$", re.IGNORECASE)
LOC_EQ_PATTERN = re.compile(r"(city|state|country|district)\s*=\s*([^\.,;]+)", re.IGNORECASE)

def slugify(s: str) -> str:
    s = s or ""
    s = "".join(ch.lower() if ch.isalnum() else "-" for ch in s)
    s = re.sub("-+", "-", s).strip("-")
    return s or "noname"

def safe_get(d: Dict[str, Any], *keys, default=""):
    try:
        cur = d
        for k in keys:
            cur = cur[k]
        return cur
    except (KeyError, TypeError):
        return default

def normalize_bool_str(v: Any) -> str:
    if isinstance(v, bool): 
        return "True" if v else "False"
    s = str(v).strip().lower()
    if s in {"true", "yes", "y", "1"}: return "True"
    if s in {"false", "no", "n", "0"}: return "False"
    return str(v)

def join_nonempty(parts: List[str], sep="; ") -> str:
    return sep.join(p for p in parts if p and str(p).strip().lower() != "nan")

# ---------------- Document building ----------------
def process_company_batch(companies_batch: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, Any]]]:
    return [flatten_company(c) for c in companies_batch]

def flatten_company(c: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    cd = c.get("CompanyDetails", {})
    name = cd.get("company_name", "")
    refno = (cd.get("company_ref_no", "") or "").strip()
    cid = str(cd.get("company_id", "")).strip()

    detail_keys = [
        "company_status","company_class","listing_status","company_category",
        "company_subcategory","industrial_classification","core_expertise",
        "industry_domain","industry_subdomain","other_expertise",
        "other_industry_domain","other_industry_subdomain",
    ]
    details = []
    for k in detail_keys:
        v = cd.get(k, "")
        if v and str(v).lower() != "nan":
            details.append(f"{k.replace('_',' ')}: {v}")

    addr_parts = [cd.get(k, "") for k in ["address","city","district","state","pincode","country"]]
    addr = join_nonempty(addr_parts, sep=", ")

    contact_parts = [
        f"email={cd.get('email','')}",
        f"poc_email={cd.get('poc_email','')}",
        f"phone={cd.get('phone','')}",
        f"website={cd.get('website','')}",
    ]
    contact = join_nonempty(contact_parts)

    products = []
    product_keys = [
        "product_name","product_description","product_type","defence_platform","platform_tech_area",
        "salient_features","hsn_code","nsn_number","items_exported","annual_production_capacity","future_expansion",
    ]
    for p in safe_get(c, "ProductsAndServices", "ProductList", default=[]):
        parts = []
        for k in product_keys:
            v = p.get(k, "")
            if v and str(v).lower() != "nan":
                if k == "items_exported":
                    v = normalize_bool_str(v)
                parts.append(f"{k.replace('_',' ')}: {v}")
        if parts:
            products.append(join_nonempty(parts))

    certs = []
    cert_keys = [
        "certification_detail","certification_type_master","certificate_number",
        "certificate_start_date","certificate_end_date","other_certification_type",
    ]
    for r in safe_get(c, "QualityAndCompliance", "CertificationsList", default=[]):
        parts = []
        for k in cert_keys:
            v = r.get(k, "")
            if v and str(v).lower() != "nan":
                parts.append(f"{k.replace('_',' ')}: {v}")
        if parts:
            certs.append(join_nonempty(parts))

    tests = []
    test_keys = ["test_details","test_nabl_accredited","test_category","test_subcategory","test_subcategory_description"]
    for t in safe_get(c, "QualityAndCompliance", "TestingCapabilitiesList", default=[]):
        parts = []
        for k in test_keys:
            v = t.get(k, "")
            if v and str(v).lower() != "nan":
                if k == "test_nabl_accredited":
                    v = normalize_bool_str(v)
                parts.append(f"{k.replace('_',' ')}: {v}")
        if parts:
            tests.append(join_nonempty(parts))

    rds = []
    rd_keys = ["rd_details","rd_nabl_accredited","rd_category","rd_subcategory"]
    for r in safe_get(c, "ResearchAndDevelopment", "RDCapabilitiesList", default=[]):
        parts = []
        for k in rd_keys:
            v = r.get(k, "")
            if v and str(v).lower() != "nan":
                if k == "rd_nabl_accredited":
                    v = normalize_bool_str(v)
                parts.append(f"{k.replace('_',' ')}: {v}")
        if parts:
            rds.append(join_nonempty(parts))

    lines = [
        f"Company Name: {name}",
        f"Company Ref No: {refno}",
        f"Company ID: {cid}",
        f"Address: {addr}",
        f"Contact: {contact}",
    ]
    if details: lines.append(f"Details: {join_nonempty(details)}")

    lines.append("Products:")
    if products: lines.extend(f"  - {p}" for p in products)
    else: lines.append("  - (none)")

    lines.append("Certifications:")
    if certs: lines.extend(f"  - {cstr}" for cstr in certs)
    else: lines.append("  - (none)")

    lines.append("Testing Capabilities:")
    if tests: lines.extend(f"  - {tstr}" for tstr in tests)
    else: lines.append("  - (none)")

    lines.append("Research & Development:")
    if rds: lines.extend(f"  - {rstr}" for rstr in rds)
    else: lines.append("  - (none)")

    hints = []
    for terms in SYNONYMS.values(): hints.extend(terms)
    lines.append("Hints: " + ", ".join(hints))
    txt = "\n".join(lines)

    meta = {
        "company_name": name, "company_ref_no": refno, "company_id": cid,
        "city": cd.get("city",""), "district": cd.get("district",""),
        "state": cd.get("state",""), "country": cd.get("country",""),
        "website": cd.get("website",""), "email": cd.get("email",""),
    }

    parts = [refno, cid, slugify(name)]
    base_id = "__".join([p for p in parts if p]) or f"company_{hashlib.md5((name+refno+cid).encode('utf-8')).hexdigest()[:12]}"
    return base_id, txt, meta

# ---------------- Indexing ----------------
def build_index(json_path: Path, db_path: Path, collection_name: str = "companies", recreate: bool = False):
    import time
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"[INDEX] FAISS-BASED INDEXING")
    print(f"{'='*60}")
    print(f"[INDEX] Loading {json_path} ...")
    print(f"[INDEX] CPU cores available: {CPU_COUNT}, using {OPTIMAL_THREADS} threads")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    companies = data.get("companies", [])
    total_companies = len(companies)
    print(f"[INDEX] Total companies to index: {total_companies:,}")

    # Check if index exists
    index_path = db_path / collection_name
    if index_path.exists() and not recreate:
        print(f"[INDEX] Index already exists at {index_path}. Use --recreate to overwrite.")
        return

    # Load embedding model
    print(f"[INDEX] Loading embedding model...")
    model = get_embedding_model()
    dimension = model.get_sentence_embedding_dimension()
    print(f"[INDEX] Embedding dimension: {dimension}")
    
    # Create FAISS store
    store = FAISSStore(dimension=dimension)

    # Process companies in chunks
    print(f"\n[STEP 1/3] Processing company data...")
    chunk_size = min(CHUNK_SIZE, max(500, len(companies) // max(1, OPTIMAL_THREADS) + 1))
    chunks = [companies[i:i + chunk_size] for i in range(0, len(companies), chunk_size)]
    print(f"[STEP 1/3] Processing {len(chunks)} chunks of size ~{chunk_size}")

    with ThreadPoolExecutor(max_workers=OPTIMAL_THREADS) as executor:
        chunk_results = list(executor.map(process_company_batch, chunks))

    # Collect all documents
    all_ids, all_docs, all_metas = [], [], []
    seen_counts = defaultdict(int)

    for chunk_result in chunk_results:
        for base_id, text, meta in chunk_result:
            seen_counts[base_id] += 1
            doc_id = base_id if seen_counts[base_id] == 1 else f"{base_id}__dup{seen_counts[base_id]}"
            all_ids.append(doc_id)
            all_docs.append(text)
            all_metas.append(meta)
    
    print(f"[STEP 1/3] ✓ Processed {len(all_docs):,} documents")

    print(f"\n[STEP 2/3] Generating embeddings...")
    
    # Generate embeddings in batches with progress tracking
    all_embeddings = []
    batch_size = 64  # Increased for better throughput
    total_batches = (len(all_docs) + batch_size - 1) // batch_size
    
    for batch_idx, i in enumerate(range(0, len(all_docs), batch_size), 1):
        batch_docs = all_docs[i:i + batch_size]
        batch_emb = model.encode(
            batch_docs,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True
        )
        all_embeddings.append(batch_emb)
        
        # Progress update every 100 docs or at end
        progress = min(i + batch_size, len(all_docs))
        if progress % 100 == 0 or progress == len(all_docs):
            pct = (progress / len(all_docs)) * 100
            elapsed = time.time() - start_time
            rate = progress / elapsed if elapsed > 0 else 0
            eta = (len(all_docs) - progress) / rate if rate > 0 else 0
            print(f"[STEP 2/3] Progress: {progress:,}/{len(all_docs):,} ({pct:.1f}%) | "
                  f"Rate: {rate:.0f} docs/sec | ETA: {eta:.0f}s", end='\r')
    
    print()  # New line after progress
    embeddings = np.vstack(all_embeddings).astype('float32')
    print(f"[STEP 2/3] ✓ Generated {len(all_embeddings):,} embedding batches")
    
    # Add to FAISS
    print(f"\n[STEP 3/3] Adding to FAISS index...")
    store.add(all_ids, all_docs, embeddings, all_metas)
    print(f"[STEP 3/3] ✓ Added {store.count():,} documents to index")
    
    # Save
    print(f"[STEP 3/3] Saving index to disk...")
    store.save(index_path)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[INDEX] ✓ INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average rate: {len(all_docs)/total_time:.0f} docs/sec")
    print(f"Collection: '{collection_name}'")
    print(f"Location: {index_path}")
    print(f"{'='*60}\n")

# ---------------- NL signal extraction & soft rerank ----------------
STOPWORDS = {
    "the","a","an","with","and","or","of","for","in","on","to","by","at","from","companies","company","list","show",
    "that","which","who","whose","is","are","having","have","has"
}

def extract_signals(query: str) -> Dict[str, Any]:
    """Extract soft constraints/signals from plain English query."""
    q = query.strip()
    ql = q.lower()
    signals: Dict[str, Any] = {
        "isos": set(),
        "want_nabl": False,
        "locations": {},
        "product_terms": set(),
        "rd_terms": set(),
        "testing_terms": set(),
        "other_keywords": set(),
        "phrases": set(),
    }

    # ISO
    for m in ISO_PATTERN.findall(ql):
        signals["isos"].add(m.replace(" ", "").upper())

    # NABL
    if "nabl" in ql or "accredited lab" in ql or "nabl-accredited" in ql:
        signals["want_nabl"] = True

    # Location
    for m in LOC_EQ_PATTERN.findall(q):
        key, val = m[0].lower(), m[1].strip()
        if key in LOCATION_KEYS:
            signals["locations"][key] = val

    m = LOC_IN_PATTERN.search(q)
    if m:
        signals["locations"]["any"] = m.group(1).strip()

    # Phrases
    words = [w for w in re.findall(r"[a-zA-Z0-9\-]+", ql) if w not in STOPWORDS and len(w) > 2]
    bigrams = [" ".join([words[i], words[i+1]]) for i in range(len(words)-1)]
    for bg in bigrams:
        if bg in {"high voltage","test lab","high-voltage"}:
            signals["phrases"].add(bg)

    # Buckets
    for w in words:
        if any(w in t.lower() for t in SYNONYMS["product"]):
            signals["product_terms"].add(w)
        elif any(w in t.lower() for t in SYNONYMS["rd"]):
            signals["rd_terms"].add(w)
        elif any(w in t.lower() for t in SYNONYMS["testing"]):
            signals["testing_terms"].add(w)
        elif any(w in t.lower() for t in SYNONYMS["certification"]):
            pass
        else:
            signals["other_keywords"].add(w)

    if "high voltage" in ql or "high-voltage" in ql:
        signals["rd_terms"].add("high voltage")
        signals["product_terms"].add("transformer")

    return signals

def _friendly_total_line(store: FAISSStore, signals, k: int = 6) -> Optional[str]:
    """Returns friendly summary line"""
    locs = signals.get("locations") or {}
    place = (locs.get("any") or "").strip()
    try:
        if not place:
            total = store.count()
        else:
            metas = store.metadatas
            p = place.lower()
            def m_ok(m):
                for f in ("city","district","state","country"):
                    if p in str(m.get(f, "")).lower():
                        return True
                return False
            total = sum(1 for m in metas if m_ok(m))
    except Exception:
        return None

    showing = min(k, total) if total else 0
    place_txt = f" in {place.title()}" if place else ""
    if total == 0:
        return f"No companies were found{place_txt}."
    if showing < total:
        return f"We found {total} companies{place_txt}. Showing the first {showing}."
    return f"We found {total} companies{place_txt}."

def soft_rerank(docs: List[str], metas: List[Dict[str, Any]], signals: Dict[str, Any]) -> List[int]:
    """Score documents by matches to signals"""
    N = len(docs)
    if N == 0: return []

    scores = [0.0] * N
    W_ISO = 5.0
    W_NABL = 3.5
    W_LOC_META = 3.0
    W_LOC_ANY = 2.0
    W_PHRASE = 2.5
    W_PROD = 2.0
    W_RD = 2.0
    W_TEST = 2.0
    W_OTHER = 0.5
    W_BASE_RANK = 0.02

    for i in range(N):
        txt_up = docs[i].upper()
        txt_lo = docs[i].lower()

        for iso in signals["isos"]:
            if iso in txt_up:
                scores[i] += W_ISO

        if signals["want_nabl"]:
            if ("RD_NABL_ACCREDITED: TRUE" in txt_up or
                "TEST_NABL_ACCREDITED: TRUE" in txt_up or
                "NABL" in txt_up):
                scores[i] += W_NABL

        for k, v in signals["locations"].items():
            if k == "any":
                if any(str(metas[i].get(f, "")).lower().find(v.lower()) >= 0 for f in LOCATION_KEYS):
                    scores[i] += W_LOC_ANY
            elif k in LOCATION_KEYS:
                if v.lower() in str(metas[i].get(k, "")).lower():
                    scores[i] += W_LOC_META

        for ph in signals["phrases"]:
            if ph in txt_lo:
                scores[i] += W_PHRASE

        for w in signals["product_terms"]:
            if w in txt_lo:
                scores[i] += W_PROD
        for w in signals["rd_terms"]:
            if w in txt_lo:
                scores[i] += W_RD
        for w in signals["testing_terms"]:
            if w in txt_lo:
                scores[i] += W_TEST
        for w in signals["other_keywords"]:
            if w in txt_lo:
                scores[i] += W_OTHER

        scores[i] += W_BASE_RANK * (N - i)

    ranked = sorted(range(N), key=lambda i: (-scores[i], i))
    return ranked

# ---------------- QA ----------------
SYS_PROMPT = (
    "You are a precise enterprise data assistant.\n"
    "Use only the provided CONTEXT (company snapshots).\n"
    "When listing companies, show company_name, company_ref_no (or id), and why it matched.\n"
    "Prefer exact fields: certifications, testing labs (NABL), R&D, products, and location.\n"
    "If filters appear (city/state/country, ISO 9001, NABL), apply them strictly.\n"
    "If not enough info, say so clearly."
)

def answer_query(
    db_path: Path,
    model_path: Path,
    ask: str,
    k: int = 6,
    max_ctx_chars: int = 12000,
    collection_name: str = "companies",
    n_threads: int = OPTIMAL_THREADS,
    n_ctx: int = 4096,
    n_batch: int = 256,
    n_gpu_layers: int = 0,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    print_json_matches: bool = False,
):
    # Load FAISS store
    index_path = db_path / collection_name
    if not index_path.exists():
        print(f"[QUERY] No index found at {index_path}.")
        print("Run: python json_rag_win.py index --json <path> --db <db> --collection companies")
        return

    store = FAISSStore.load(index_path)
    
    if store.count() == 0:
        print(f"[QUERY] Index is empty.")
        return

    # Embed query
    model = get_embedding_model()
    query_embedding = model.encode([ask], convert_to_numpy=True)[0].astype('float32')

    # Search
    pool = max(k * 4, 50)
    res = store.query(query_embedding, n_results=pool)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    ids = res["ids"][0]

    # Extract signals and rerank
    signals = extract_signals(ask)
    ranked = soft_rerank(docs, metas, signals)
    keep = ranked[:k] if ranked else list(range(min(k, len(docs))))

    # Build context
    selected, running = [], 0
    for i in keep:
        snap = f"- ID: {ids[i]}\n  NAME: {metas[i].get('company_name','')}\n  META: {metas[i]}\n  TEXT:\n{docs[i]}\n"
        if running + len(snap) > max_ctx_chars:
            break
        selected.append(snap)
        running += len(snap)
    context = "CONTEXT:\n" + ("\n".join(selected) if selected else "(No results)")

    if print_json_matches:
        out_list = []
        for i in keep[:10]:
            out_list.append({
                "id": ids[i],
                "company_name": metas[i].get("company_name",""),
                "city": metas[i].get("city",""),
                "state": metas[i].get("state",""),
                "country": metas[i].get("country",""),
                "email": metas[i].get("email",""),
                "website": metas[i].get("website",""),
            })
        print(json.dumps({"matches": out_list}, ensure_ascii=False, indent=2))

    # LLM
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
        use_mmap=True,
        use_mlock=False,
        n_batch=n_batch,
    )

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"{ask}\n\n{context}"},
    ]
    out = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )
    text = out["choices"][0]["message"]["content"].strip()

    print("\n=== ANSWER ===\n")
    
    lead = None
    if LIST_TOTAL_Q.search(ask):
        try:
            lead = _friendly_total_line(store, signals, k)
        except Exception:
            lead = None
    if lead:
        print(lead)
        print()

    print(text)
    del llm
    gc.collect()

# ---------------- CLI ----------------
def main():
    global BATCH_SIZE
    if sys.platform == "win32":
        try:
            import psutil
            p = psutil.Process()
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="JSON RAG for Windows with FAISS")
    sub = ap.add_subparsers(dest="cmd", required=True)

    api = sub.add_parser("index", help="Index the JSON into FAISS")
    api.add_argument("--json", required=True, type=Path)
    api.add_argument("--db", required=True, type=Path)
    api.add_argument("--collection", default="companies")
    api.add_argument("--recreate", action="store_true")
    api.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    apq = sub.add_parser("query", help="Ask a question")
    apq.add_argument("--db", required=True, type=Path)
    apq.add_argument("--model", required=True, type=Path)
    apq.add_argument("--ask", required=True, type=str)
    apq.add_argument("--k", type=int, default=8)
    apq.add_argument("--collection", default="companies")
    apq.add_argument("--max-ctx-chars", type=int, default=12000)
    apq.add_argument("--temperature", type=float, default=0.2)
    apq.add_argument("--top-p", type=float, default=0.9)
    apq.add_argument("--repeat-penalty", type=float, default=1.1)
    apq.add_argument("--n-threads", type=int, default=OPTIMAL_THREADS)
    apq.add_argument("--n-ctx", type=int, default=4096)
    apq.add_argument("--n-batch", type=int, default=1024)
    apq.add_argument("--n-gpu-layers", type=int, default=0)
    apq.add_argument("--json", dest="print_json_matches", action="store_true")

    args = ap.parse_args()

    if getattr(args, "batch_size", None) is not None and args.cmd == "index":
        BATCH_SIZE = args.batch_size

    if args.cmd == "index":
        build_index(args.json, args.db, collection_name=args.collection, recreate=args.recreate)
    elif args.cmd == "query":
        answer_query(
            args.db, args.model, args.ask,
            k=args.k,
            max_ctx_chars=args.max_ctx_chars,
            collection_name=args.collection,
            n_threads=args.n_threads,
            n_ctx=args.n_ctx,
            n_batch=args.n_batch,
            n_gpu_layers=args.n_gpu_layers,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            print_json_matches=args.print_json_matches,
        )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
