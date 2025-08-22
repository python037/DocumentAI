# app.py

# =======================
# ======= Imports =======
# =======================

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
from langchain_experimental.text_splitter import \
    SemanticChunker  # (optional, for semantic chunking)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PIL import Image

# =============================================================================
# App and Global Configuration
# =============================================================================

load_dotenv(r"C:\ML\LU-LiveClasses\DocumentAI\.env")
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