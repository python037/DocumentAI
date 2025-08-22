# app.py

import io
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import google.generativeai as genai
import pytesseract
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import \
    RecursiveCharacterTextSplitter  # (default text splitter)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_experimental.text_splitter import \
    SemanticChunker  # (optional, for semantic chunking)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PIL import Image

# =============================================================================
# App and Global Configuration
# =============================================================================

load_dotenv()
app = Flask(__name__)
CORS(app)

# --- Paths ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "input_data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Embedding Backend Setting ---
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "gemini").strip().lower()  # "minilm" or "gemini"
VECTOR_DIR = os.path.join(DATA_DIR, f"faiss_{EMBEDDING_BACKEND}")
os.makedirs(VECTOR_DIR, exist_ok=True)

# --- Gemini API Key for Chat and (optional) Embeddings ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- Optional Tesseract Path for Windows ---
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# --- Chunking Defaults ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNKING_MODE_DEFAULT = os.getenv("CHUNKING_MODE", "semantic").strip().lower()  # "recursive" or "semantic"

# --- SemanticChunker tuning (used only when CHUNKING_MODE=semantic) ---
SEMANTIC_BREAKPOINT_TYPE = os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile").strip().lower()
# Typically 95 works well for "percentile" type
try:
    SEMANTIC_BREAKPOINT_AMOUNT = float(os.getenv("SEMANTIC_BREAKPOINT_AMOUNT", "95"))
except ValueError:
    SEMANTIC_BREAKPOINT_AMOUNT = 95.0

# --- Chat Defaults ---
TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))

# --- Thread-safety for Vector Store Updates ---
INDEX_LOCK = threading.Lock()

# --- In-memory Chat Sessions: session_id -> list of {role, content} ---
CHAT_SESSIONS: Dict[str, List[Dict[str, str]]] = {}

# --- Global Embeddings Instance (lazy init) ---
_EMBEDDINGS: Optional[Embeddings] = None


# =============================================================================
# Utility and Core Functions
# =============================================================================

def get_embeddings() -> Embeddings:
    """
    Return a shared embeddings instance based on EMBEDDING_BACKEND.
    - "minilm": sentence-transformers/all-MiniLM-L6-v2 (CPU-friendly)
    - "gemini": Google text-embedding-004 (requires GOOGLE_API_KEY)
    """
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    if EMBEDDING_BACKEND == "gemini":
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini embeddings.")
        _EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    else:
        # Default local embedding
        _EMBEDDINGS = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDINGS

def load_vector_store(allow_create_empty: bool = True) -> Optional[FAISS]:
    """
    Load the FAISS vector store from disk. If not found and allow_create_empty is False,
    returns None. Otherwise, returns a ready-to-use FAISS store (or None if empty).
    """
    embeddings = get_embeddings()
    index_file = os.path.join(VECTOR_DIR, "index.faiss")
    store_file = os.path.join(VECTOR_DIR, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(store_file):
        # allow_dangerous_deserialization required when loading pickled docstore
        return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)

    if not allow_create_empty:
        return None
    return None

def save_vector_store(store: FAISS) -> None:
    """
    Persist the FAISS vector store to disk at VECTOR_DIR.
    """
    store.save_local(VECTOR_DIR)

def build_text_splitter(
    mode: str = CHUNKING_MODE_DEFAULT,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """
    Build and return a text splitter based on the selected mode.
    - recursive: RecursiveCharacterTextSplitter
    - semantic: SemanticChunker (requires embeddings)
    """
    mode = (mode or "recursive").strip().lower()
    if mode == "semantic":
        embeddings = get_embeddings()
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_AMOUNT,
        )
    # default fallback: recursive
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def chunk_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    chunking_mode: str = CHUNKING_MODE_DEFAULT,
) -> List[Document]:
    """
    Split a list of Documents into chunks using the configured splitter.
    Supports:
    - recursive (default)
    - semantic (optional)
    """
    splitter = build_text_splitter(mode=chunking_mode, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def add_documents_to_index(
    docs: List[Document],
    chunking_mode: str = CHUNKING_MODE_DEFAULT,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Tuple[int, int]:
    """
    Add Documents to FAISS index (create if needed), then persist.
    Returns (num_original_docs, num_chunks_added).
    """
    if not docs:
        return 0, 0

    # Chunk first
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap, chunking_mode=chunking_mode)

    embeddings = get_embeddings()

    with INDEX_LOCK:
        store = load_vector_store(allow_create_empty=True)
        if store is None:
            store = FAISS.from_documents(chunks, embeddings)
        else:
            store.add_documents(chunks)
        save_vector_store(store)

    return len(docs), len(chunks)

def pdf_to_documents(pdf_path: str, source_id: Optional[str] = None) -> List[Document]:
    """
    Extract text from a PDF using PyPDFLoader (one Document per page).
    Adds simple metadata for citations (source, page).
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        # Attach/standardize metadata
        for i, d in enumerate(pages):
            d.metadata = d.metadata or {}
            d.metadata.update({
                "source": os.path.basename(pdf_path),
                "doc_id": source_id or os.path.splitext(os.path.basename(pdf_path))[0],
                "page": d.metadata.get("page", i + 1)
            })
        return pages
    except Exception as e:
        # print(f"[WARN] PyPDFLoader failed on {pdf_path}: {e}")
        return []

def ocr_image_bytes(image_bytes: bytes, source_name: str) -> Document:
    """
    OCR a single image (bytes) into a Document with metadata for citation.
    """
    img = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(img)
    return Document(
        page_content=text.strip(),
        metadata={
            "source": source_name,
            "doc_id": os.path.splitext(os.path.basename(source_name))[0],
            "page": None,
            "kind": "image_ocr"
        }
    )

def ocr_pdf_pages(pdf_path: str, dpi: int = 200) -> List[Document]:
    """
    OCR a PDF by rasterizing each page and running Tesseract OCR.
    Use when text extraction fails (e.g., scanned PDFs).
    """
    out_docs: List[Document] = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
                out_docs.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "doc_id": os.path.splitext(os.path.basename(pdf_path))[0],
                        "page": page_index + 1,
                        "kind": "pdf_ocr"
                    }
                ))
    except Exception as e:
        print(f"[ERROR] OCR on PDF failed for {pdf_path}: {e}")
    return out_docs

def txt_to_document(text_bytes: bytes, filename: str) -> Document:
    """
    Convert a text file (bytes) to a single Document.
    """
    try:
        content = text_bytes.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        content = text_bytes.decode("latin-1", errors="ignore")
    return Document(
        page_content=content.strip(),
        metadata={
            "source": filename,
            "doc_id": os.path.splitext(os.path.basename(filename))[0],
            "page": None,
            "kind": "text"
        }
    )
