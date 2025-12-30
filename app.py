"""
Chat with PDF - Advanced AI-Powered PDF Question Answering Application
Author: AI Assistant
Description: A comprehensive Streamlit application for uploading, processing, and chatting with PDF documents
using advanced NLP, vector embeddings, and LLM integration.
"""

import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import os
import sqlite3
import bcrypt
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import time
import re

# AI/NLP Libraries
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.embeddings import HuggingFaceEmbeddings
import spacy

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_REQUESTS_PER_MINUTE = 30
DB_PATH = "chat_pdf.db"

# Initialize spaCy model for NER (download with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("spaCy model not found. Entity recognition will be disabled.")
    nlp = None


# ==================== DATABASE SETUP ====================
def init_database():
    """Initialize SQLite database for user management and chat history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Chat history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT,
            pdf_name TEXT,
            message_type TEXT,
            message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Usage analytics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usage_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


# ==================== USER AUTHENTICATION ====================
class UserAuth:
    """Handle user authentication and session management."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    @staticmethod
    def register_user(username: str, password: str) -> Tuple[bool, str]:
        """Register a new user."""
        if len(username) < 3 or len(password) < 6:
            return False, "Username must be 3+ chars, password 6+ chars"
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            password_hash = UserAuth.hash_password(password)
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                         (username, password_hash))
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            logger.info(f"User registered: {username}")
            return True, f"User {username} registered successfully!"
        except sqlite3.IntegrityError:
            return False, "Username already exists"
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False, f"Registration failed: {str(e)}"
    
    @staticmethod
    def login_user(username: str, password: str) -> Tuple[bool, Optional[int], str]:
        """Authenticate a user."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            conn.close()
            
            if result and UserAuth.verify_password(password, result[1]):
                logger.info(f"User logged in: {username}")
                return True, result[0], "Login successful!"
            return False, None, "Invalid username or password"
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False, None, f"Login failed: {str(e)}"


# ==================== PDF PROCESSING ====================
class PDFProcessor:
    """Handle PDF upload, extraction, and processing."""
    
    @staticmethod
    def validate_pdf(file) -> Tuple[bool, str]:
        """Validate uploaded PDF file."""
        if file.size > MAX_FILE_SIZE:
            return False, f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB"
        
        if not file.name.lower().endswith('.pdf'):
            return False, "Only PDF files are supported"
        
        return True, "Valid PDF"
    
    @staticmethod
    def extract_text_pymupdf(pdf_bytes: bytes) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PyMuPDF extraction error: {e}")
            return ""
    
    @staticmethod
    def extract_text_pdfplumber(pdf_bytes: bytes) -> str:
        """Extract text from PDF using pdfplumber (fallback)."""
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            logger.error(f"pdfplumber extraction error: {e}")
            return ""
    
    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        """Extract text using primary and fallback methods."""
        text = PDFProcessor.extract_text_pymupdf(pdf_bytes)
        if not text.strip():
            logger.warning("PyMuPDF failed, trying pdfplumber")
            text = PDFProcessor.extract_text_pdfplumber(pdf_bytes)
        return text
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy."""
        if not nlp or not text:
            return {}
        
        try:
            doc = nlp(text[:100000])  # Limit for performance
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                if ent.text not in entities[ent.label_]:
                    entities[ent.label_].append(ent.text)
            return entities
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return {}
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """Extract keywords using simple frequency analysis."""
        if not text:
            return []
        
        # Simple keyword extraction (frequency-based)
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'their', 'which'}
        words = [w for w in words if w not in stop_words]
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]


# ==================== VECTOR STORE & EMBEDDINGS ====================
class VectorStoreManager:
    """Manage vector embeddings and semantic search."""
    
    def __init__(self):
        self.embedding_model = None
        self.vectorstore = None
        self.documents = []
    
    def initialize_embeddings(self):
        """Initialize sentence transformer model."""
        if not self.embedding_model:
            with st.spinner("Loading embedding model..."):
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("Embedding model loaded")
    
    def create_vectorstore(self, text: str, pdf_name: str) -> bool:
        """Create vector store from PDF text."""
        try:
            self.initialize_embeddings()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )
            
            chunks = text_splitter.split_text(text)
            self.documents = [
                Document(page_content=chunk, metadata={"source": pdf_name, "chunk": i})
                for i, chunk in enumerate(chunks)
            ]
            
            # Create FAISS vectorstore
            self.vectorstore = LangchainFAISS.from_documents(
                self.documents,
                self.embedding_model
            )
            
            logger.info(f"Vector store created with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Vector store creation error: {e}")
            st.error(f"Failed to create vector store: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Perform similarity search."""
        if not self.vectorstore:
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return []


# ==================== LLM & CHAT ====================
class ChatManager:
    """Manage chat interactions with LLM."""
    
    def __init__(self, vectorstore_manager: VectorStoreManager):
        self.vectorstore_manager = vectorstore_manager
        self.llm = None
        self.qa_chain = None
        self.memory = None
    
    def initialize_llm(self, temperature: float = 0.7, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI LLM."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            return False
        
        try:
            self.llm = ChatOpenAI(
                model_name=model,
                temperature=temperature,
                openai_api_key=api_key
            )
            
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            if self.vectorstore_manager.vectorstore:
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vectorstore_manager.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    memory=self.memory,
                    return_source_documents=True
                )
            
            logger.info("LLM initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            st.error(f"Failed to initialize LLM: {str(e)}")
            return False
    
    def get_response(self, question: str) -> Tuple[str, List[Document]]:
        """Get response from LLM based on question."""
        if not self.qa_chain:
            return "Please upload a PDF and initialize the chat first.", []
        
        try:
            result = self.qa_chain({"question": question})
            answer = result.get("answer", "No answer generated.")
            source_docs = result.get("source_documents", [])
            return answer, source_docs
            
        except Exception as e:
            logger.error(f"Chat response error: {e}")
            return f"Error generating response: {str(e)}", []
    
    def generate_summary(self, text: str, max_length: int = 500) -> str:
        """Generate summary of PDF content."""
        if not self.llm:
            return "LLM not initialized"
        
        try:
            # Truncate text if too long
            truncated_text = text[:4000]
            
            prompt = f"""Please provide a concise summary of the following text in about {max_length} characters:

{truncated_text}

Summary:"""
            
            response = self.llm.predict(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return f"Error generating summary: {str(e)}"


# ==================== DATABASE HELPERS ====================
def save_chat_message(user_id: int, session_id: str, pdf_name: str, 
                     message_type: str, message: str):
    """Save chat message to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (user_id, session_id, pdf_name, message_type, message) VALUES (?, ?, ?, ?, ?)",
            (user_id, session_id, pdf_name, message_type, message)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving chat message: {e}")


def log_usage(user_id: int, action: str, details: str = ""):
    """Log user actions for analytics."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO usage_analytics (user_id, action, details) VALUES (?, ?, ?)",
            (user_id, action, details)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error logging usage: {e}")


def get_chat_history(user_id: int, session_id: str) -> List[Dict]:
    """Retrieve chat history for a session."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT message_type, message, timestamp FROM chat_history WHERE user_id = ? AND session_id = ? ORDER BY timestamp",
            (user_id, session_id)
        )
        results = cursor.fetchall()
        conn.close()
        
        return [{"type": r[0], "message": r[1], "timestamp": r[2]} for r in results]
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        return []


# ==================== RATE LIMITING ====================
class RateLimiter:
    """Simple rate limiter to prevent abuse."""
    
    def __init__(self):
        if 'rate_limit_data' not in st.session_state:
            st.session_state.rate_limit_data = {}
    
    def check_limit(self, user_id: int, max_requests: int = MAX_REQUESTS_PER_MINUTE) -> bool:
        """Check if user has exceeded rate limit."""
        current_time = time.time()
        
        if user_id not in st.session_state.rate_limit_data:
            st.session_state.rate_limit_data[user_id] = []
        
        # Remove old requests (older than 1 minute)
        st.session_state.rate_limit_data[user_id] = [
            t for t in st.session_state.rate_limit_data[user_id]
            if current_time - t < 60
        ]
        
        if len(st.session_state.rate_limit_data[user_id]) >= max_requests:
            return False
        
        st.session_state.rate_limit_data[user_id].append(current_time)
        return True


# ==================== UI COMPONENTS ====================
def render_login_page():
    """Render login/signup page."""
    st.title("🔐 Chat with PDF - Login")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login to Your Account")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_btn"):
            if username and password:
                success, user_id, message = UserAuth.login_user(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("Please enter both username and password")
    
    with tab2:
        st.subheader("Create New Account")
        new_username = st.text_input("Username", key="signup_username")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
        
        if st.button("Sign Up", key="signup_btn"):
            if new_username and new_password and confirm_password:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = UserAuth.register_user(new_username, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.warning("Please fill all fields")


def render_sidebar():
    """Render sidebar with settings and controls."""
    st.sidebar.title("⚙️ Settings")
    
    # User info
    st.sidebar.info(f"👤 Logged in as: **{st.session_state.username}**")
    
    if st.sidebar.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # AI Parameters
    st.sidebar.subheader("🤖 AI Parameters")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1,
                                   help="Higher = more creative, Lower = more focused")
    model_choice = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"],
                                       help="Select OpenAI model")
    
    st.sidebar.markdown("---")
    
    # App Info
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        "This app allows you to upload PDFs and chat with them using AI. "
        "Upload a PDF, ask questions, and get intelligent answers based on the content."
    )
    
    return temperature, model_choice


def export_chat_history(chat_messages: List[Dict]) -> str:
    """Export chat history as text."""
    export_text = "Chat History Export\n"
    export_text += "=" * 50 + "\n\n"
    
    for msg in chat_messages:
        timestamp = msg.get('timestamp', 'N/A')
        msg_type = msg['type'].upper()
        message = msg['message']
        export_text += f"[{timestamp}] {msg_type}:\n{message}\n\n"
    
    return export_text


def render_main_app():
    """Render main application interface."""
    st.title("📄 Chat with PDF")
    st.markdown("Upload your PDF and start asking questions!")
    
    # Sidebar
    temperature, model_choice = render_sidebar()
    
    # Initialize session state
    if 'vector_manager' not in st.session_state:
        st.session_state.vector_manager = VectorStoreManager()
    
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = ChatManager(st.session_state.vector_manager)
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    if 'current_pdf_name' not in st.session_state:
        st.session_state.current_pdf_name = None
    
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    
    if 'rate_limiter' not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()
    
    # File upload section
    st.subheader("📤 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        process_btn = st.button("🔄 Process PDF", disabled=uploaded_file is None)
    
    with col2:
        if st.session_state.pdf_processed:
            summary_btn = st.button("📝 Generate Summary")
        else:
            summary_btn = False
    
    with col3:
        if st.session_state.pdf_processed:
            analyze_btn = st.button("🔍 Analyze Content")
        else:
            analyze_btn = False
    
    # Process PDF
    if process_btn and uploaded_file:
        # Validate PDF
        is_valid, message = PDFProcessor.validate_pdf(uploaded_file)
        if not is_valid:
            st.error(message)
        else:
            with st.spinner("Processing PDF..."):
                try:
                    # Extract text
                    pdf_bytes = uploaded_file.read()
                    text = PDFProcessor.extract_text(pdf_bytes)
                    
                    if not text.strip():
                        st.error("Could not extract text from PDF. The file may be image-based or corrupted.")
                    else:
                        st.session_state.pdf_text = text
                        st.session_state.current_pdf_name = uploaded_file.name
                        
                        # Create vector store
                        success = st.session_state.vector_manager.create_vectorstore(
                            text, uploaded_file.name
                        )
                        
                        if success:
                            # Initialize LLM
                            llm_success = st.session_state.chat_manager.initialize_llm(
                                temperature=temperature,
                                model=model_choice
                            )
                            
                            if llm_success:
                                st.session_state.pdf_processed = True
                                st.success(f"✅ PDF processed successfully! ({len(text)} characters extracted)")
                                log_usage(st.session_state.user_id, "pdf_upload", uploaded_file.name)
                            else:
                                st.error("Failed to initialize LLM. Check your API key.")
                        else:
                            st.error("Failed to create vector store.")
                
                except Exception as e:
                    logger.error(f"PDF processing error: {e}")
                    st.error(f"Error processing PDF: {str(e)}")
    
    # Generate Summary
    if summary_btn:
        with st.spinner("Generating summary..."):
            summary = st.session_state.chat_manager.generate_summary(st.session_state.pdf_text)
            st.subheader("📝 Document Summary")
            st.info(summary)
            log_usage(st.session_state.user_id, "generate_summary", st.session_state.current_pdf_name)
    
    # Analyze Content
    if analyze_btn:
        with st.spinner("Analyzing content..."):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("🔑 Keywords")
                keywords = PDFProcessor.extract_keywords(st.session_state.pdf_text)
                st.write(", ".join(keywords))
            
            with col_b:
                st.subheader("🏷️ Named Entities")
                entities = PDFProcessor.extract_entities(st.session_state.pdf_text)
                if entities:
                    for entity_type, entity_list in list(entities.items())[:5]:
                        st.write(f"**{entity_type}**: {', '.join(entity_list[:5])}")
                else:
                    st.write("Entity recognition not available")
            
            log_usage(st.session_state.user_id, "analyze_content", st.session_state.current_pdf_name)
    
    st.markdown("---")
    
    # Chat Interface
    if st.session_state.pdf_processed:
        st.subheader("💬 Chat with Your PDF")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_messages:
                if msg['type'] == 'user':
                    st.chat_message("user").write(msg['message'])
                else:
                    st.chat_message("assistant").write(msg['message'])
        
        # Chat input
        user_question = st.chat_input("Ask a question about your PDF...")
        
        if user_question:
            # Rate limiting
            if not st.session_state.rate_limiter.check_limit(st.session_state.user_id):
                st.error("Rate limit exceeded. Please wait a moment before asking more questions.")
            else:
                # Add user message
                st.session_state.chat_messages.append({
                    'type': 'user',
                    'message': user_question,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Save to database
                save_chat_message(
                    st.session_state.user_id,
                    st.session_state.session_id,
                    st.session_state.current_pdf_name,
                    'user',
                    user_question
                )
                
                # Get AI response
                with st.spinner("Thinking..."):
                    answer, source_docs = st.session_state.chat_manager.get_response(user_question)
                    
                    # Add assistant message
                    st.session_state.chat_messages.append({
                        'type': 'assistant',
                        'message': answer,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Save to database
                    save_chat_message(
                        st.session_state.user_id,
                        st.session_state.session_id,
                        st.session_state.current_pdf_name,
                        'assistant',
                        answer
                    )
                    
                    log_usage(st.session_state.user_id, "chat_query", user_question)
                
                st.rerun()
        
        # Chat controls
        st.markdown("---")
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_messages = []
                st.session_state.chat_manager.memory.clear()
                st.rerun()
        
        with col_c2:
            if st.button("🔄 Reset Session"):
                st.session_state.pdf_processed = False
                st.session_state.chat_messages = []
                st.session_state.pdf_text = ""
                st.session_state.current_pdf_name = None
                st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.rerun()
        
        with col_c3:
            if st.button("💾 Export Chat"):
                export_text = export_chat_history(st.session_state.chat_messages)
                st.download_button(
                    label="Download Chat History",
                    data=export_text,
                    file_name=f"chat_history_{st.session_state.session_id}.txt",
                    mime="text/plain"
                )
    
    else:
        st.info("
