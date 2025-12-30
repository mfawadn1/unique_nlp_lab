# app.py
"""
Chat with PDF Application
A comprehensive local PDF chatbot using open-source libraries
Supports multiple PDFs, embeddings, vector search, Q&A, and advanced features
"""

import streamlit as st
import os
import io
import json
import time
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import numpy as np

# PDF Processing
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

# OCR Support
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# NLP and Embeddings
from sentence_transformers import SentenceTransformer
import faiss

# Transformers for Q&A
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoModelForSeq2SeqLM

# Text Processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

# Additional utilities
import pandas as pd
import pickle


# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data packages"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)


# Initialize NLTK
download_nltk_data()


# Load spaCy model
@st.cache_resource
def load_spacy_model():
    """Load spaCy model for NER and advanced processing"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        return None


class PDFProcessor:
    """Handles PDF text extraction and preprocessing"""
    
    def __init__(self, use_ocr: bool = False):
        self.use_ocr = use_ocr and OCR_AVAILABLE
        
    def extract_text_from_pdf(self, pdf_file) -> Dict[str, Any]:
        """Extract text from PDF file with metadata"""
        try:
            pdf_reader = PdfReader(pdf_file)
            
            metadata = {
                'num_pages': len(pdf_reader.pages),
                'title': pdf_reader.metadata.get('/Title', 'Unknown') if pdf_reader.metadata else 'Unknown',
                'author': pdf_reader.metadata.get('/Author', 'Unknown') if pdf_reader.metadata else 'Unknown',
                'subject': pdf_reader.metadata.get('/Subject', '') if pdf_reader.metadata else '',
                'creator': pdf_reader.metadata.get('/Creator', '') if pdf_reader.metadata else '',
            }
            
            pages_text = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                # Try OCR if text extraction fails and OCR is enabled
                if (not text or len(text.strip()) < 50) and self.use_ocr:
                    text = self._ocr_page(pdf_file, page_num)
                
                pages_text.append({
                    'page_num': page_num + 1,
                    'text': text
                })
            
            return {
                'metadata': metadata,
                'pages': pages_text,
                'full_text': '\n\n'.join([p['text'] for p in pages_text])
            }
        except Exception as e:
            st.error(f"Error extracting PDF: {str(e)}")
            return None
    
    def _ocr_page(self, pdf_file, page_num: int) -> str:
        """OCR a specific page"""
        try:
            pdf_file.seek(0)
            images = convert_from_bytes(pdf_file.read(), first_page=page_num+1, last_page=page_num+1)
            if images:
                text = pytesseract.image_to_string(images[0])
                return text
        except Exception as e:
            st.warning(f"OCR failed for page {page_num + 1}: {str(e)}")
        return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\(\)]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        # Sentence-aware chunking
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_words = word_tokenize(sentence)
            sentence_size = len(sentence_words)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'size': current_size
                })
                
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(current_chunk[-3:]):
                    overlap_size += len(word_tokenize(sent))
                    overlap_sentences.insert(0, sent)
                    if overlap_size >= overlap:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'size': current_size
            })
        
        return chunks


class EmbeddingManager:
    """Manages embeddings and vector database"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = self._load_embedding_model(model_name)
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
    @st.cache_resource
    def _load_embedding_model(_self, model_name: str):
        """Load sentence transformer model"""
        return SentenceTransformer(model_name)
    
    def create_embeddings(self, chunks: List[Dict[str, Any]], pdf_name: str, progress_callback=None):
        """Create embeddings for text chunks"""
        texts = [chunk['text'] for chunk in chunks]
        
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
            
            if progress_callback:
                progress_callback((i + len(batch)) / len(texts))
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Store chunks with metadata
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk['text'])
            self.chunk_metadata.append({
                'pdf_name': pdf_name,
                'chunk_index': i,
                'size': chunk['size']
            })
        
        # Build or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        self.index.add(embeddings)
        
        return embeddings
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'text': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx],
                    'score': float(dist)
                })
        
        return results
    
    def clear(self):
        """Clear all embeddings and index"""
        self.index = None
        self.chunks = []
        self.chunk_metadata = []


class QuestionAnsweringEngine:
    """Handles question answering using local models"""
    
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        self.qa_pipeline = self._load_qa_model(model_name)
        self.summarization_pipeline = self._load_summarization_model()
        self.nlp = load_spacy_model()
        
    @st.cache_resource
    def _load_qa_model(_self, model_name: str):
        """Load question answering model"""
        try:
            return pipeline("question-answering", model=model_name, tokenizer=model_name)
        except Exception as e:
            st.error(f"Error loading Q&A model: {str(e)}")
            return None
    
    @st.cache_resource
    def _load_summarization_model(_self):
        """Load summarization model"""
        try:
            return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
        except Exception as e:
            st.warning(f"Summarization model not available: {str(e)}")
            return None
    
    def answer_question(self, question: str, context_chunks: List[Dict[str, Any]], max_context_length: int = 2000) -> Dict[str, Any]:
        """Generate answer from context chunks"""
        if not self.qa_pipeline:
            return {
                'answer': "Q&A model not available",
                'confidence': 0.0,
                'source_chunk': None
            }
        
        # Combine top chunks as context
        context = ' '.join([chunk['text'] for chunk in context_chunks[:3]])
        
        # Truncate context if too long
        if len(context) > max_context_length:
            context = context[:max_context_length]
        
        try:
            result = self.qa_pipeline(question=question, context=context)
            
            # Find which chunk contains the answer
            answer_text = result['answer']
            source_chunk = None
            for chunk in context_chunks:
                if answer_text.lower() in chunk['text'].lower():
                    source_chunk = chunk
                    break
            
            return {
                'answer': result['answer'],
                'confidence': result['score'],
                'source_chunk': source_chunk,
                'context': context[:500] + '...' if len(context) > 500 else context
            }
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 0.0,
                'source_chunk': None
            }
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Summarize text"""
        if not self.summarization_pipeline:
            # Fallback: return first few sentences
            sentences = sent_tokenize(text)
            return ' '.join(sentences[:3])
        
        try:
            # Truncate input if too long
            max_input = 1024
            if len(text) > max_input:
                text = text[:max_input]
            
            summary = self.summarization_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            sentences = sent_tokenize(text)
            return ' '.join(sentences[:3])
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text[:1000000])  # Limit text size
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis"""
        # Simple keyword-based sentiment
        positive_words = {'good', 'great', 'excellent', 'positive', 'benefit', 'advantage', 'success'}
        negative_words = {'bad', 'poor', 'negative', 'problem', 'issue', 'fail', 'disadvantage'}
        
        words = set(word_tokenize(text.lower()))
        
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        
        if pos_count > neg_count:
            return "Positive"
        elif neg_count > pos_count:
            return "Negative"
        else:
            return "Neutral"


class ChatManager:
    """Manages chat history and context"""
    
    def __init__(self):
        self.history = []
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to chat history"""
        self.history.append({
            'role': role,
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def get_history(self, last_n: int = None) -> List[Dict]:
        """Get chat history"""
        if last_n:
            return self.history[-last_n:]
        return self.history
    
    def clear_history(self):
        """Clear chat history"""
        self.history = []
    
    def export_history(self) -> str:
        """Export chat history as JSON"""
        return json.dumps(self.history, indent=2)
    
    def get_context_for_query(self, query: str, last_n: int = 3) -> str:
        """Build context from recent chat history"""
        recent = self.get_history(last_n * 2)
        context_parts = []
        
        for msg in recent:
            if msg['role'] == 'user':
                context_parts.append(f"Question: {msg['content']}")
            elif msg['role'] == 'assistant':
                context_parts.append(f"Answer: {msg['content'][:200]}")
        
        return '\n'.join(context_parts)


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    
    if 'embedding_manager' not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager()
    
    if 'qa_engine' not in st.session_state:
        st.session_state.qa_engine = QuestionAnsweringEngine()
    
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    
    if 'processed_pdfs' not in st.session_state:
        st.session_state.processed_pdfs = {}
    
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""


def process_uploaded_pdfs(uploaded_files, chunk_size: int, use_ocr: bool):
    """Process uploaded PDF files"""
    st.session_state.pdf_processor = PDFProcessor(use_ocr=use_ocr)
    
    all_chunks = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Extract text
        pdf_data = st.session_state.pdf_processor.extract_text_from_pdf(uploaded_file)
        
        if pdf_data:
            # Store PDF metadata
            st.session_state.processed_pdfs[uploaded_file.name] = pdf_data['metadata']
            
            # Clean and chunk text
            cleaned_text = st.session_state.pdf_processor.clean_text(pdf_data['full_text'])
            chunks = st.session_state.pdf_processor.chunk_text(cleaned_text, chunk_size=chunk_size)
            
            # Create embeddings
            st.session_state.embedding_manager.create_embeddings(
                chunks, 
                uploaded_file.name,
                progress_callback=lambda p: progress_bar.progress((idx + p) / len(uploaded_files))
            )
            
            all_chunks.extend(chunks)
        
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    status_text.text(f"✓ Processed {len(uploaded_files)} PDF(s) with {len(all_chunks)} chunks")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return len(all_chunks)


def handle_query(query: str, query_type: str, top_k: int):
    """Handle user query and generate response"""
    # Search for relevant chunks
    relevant_chunks = st.session_state.embedding_manager.search(query, top_k=top_k)
    
    if not relevant_chunks:
        response = "No relevant information found in the uploaded PDFs."
        st.session_state.chat_manager.add_message('assistant', response)
        return response
    
    # Different handling based on query type
    if query_type == "Direct Answer":
        result = st.session_state.qa_engine.answer_question(query, relevant_chunks)
        response = result['answer']
        confidence = result['confidence']
        
        # Add source information
        if result['source_chunk']:
            source_info = f"\n\n*Source: {result['source_chunk']['metadata']['pdf_name']} (Confidence: {confidence:.2%})*"
            response += source_info
    
    elif query_type == "Summary":
        # Combine top chunks and summarize
        combined_text = ' '.join([chunk['text'] for chunk in relevant_chunks[:3]])
        response = st.session_state.qa_engine.summarize_text(combined_text)
        response = f"**Summary:**\n\n{response}"
    
    elif query_type == "Entity Extraction":
        # Extract entities from top chunk
        entities = st.session_state.qa_engine.extract_entities(relevant_chunks[0]['text'])
        if entities:
            entity_summary = {}
            for ent in entities:
                if ent['label'] not in entity_summary:
                    entity_summary[ent['label']] = []
                if ent['text'] not in entity_summary[ent['label']]:
                    entity_summary[ent['label']].append(ent['text'])
            
            response = "**Entities Found:**\n\n"
            for label, texts in entity_summary.items():
                response += f"**{label}:** {', '.join(texts[:5])}\n"
        else:
            response = "No entities found in the relevant sections."
    
    elif query_type == "Sentiment Analysis":
        sentiment = st.session_state.qa_engine.analyze_sentiment(relevant_chunks[0]['text'])
        response = f"**Sentiment Analysis:**\n\nThe relevant section appears to have a **{sentiment}** sentiment."
    
    else:  # Keyword Search
        response = "**Relevant Excerpts:**\n\n"
        for i, chunk in enumerate(relevant_chunks[:3], 1):
            excerpt = chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text']
            response += f"{i}. {excerpt}\n\n"
            response += f"*From: {chunk['metadata']['pdf_name']}*\n\n"
    
    # Store in chat history
    st.session_state.chat_manager.add_message('assistant', response, {
        'query_type': query_type,
        'chunks_used': len(relevant_chunks)
    })
    
    return response


def display_pdf_metadata():
    """Display metadata of processed PDFs"""
    if st.session_state.processed_pdfs:
        st.sidebar.subheader("📄 Processed PDFs")
        
        for pdf_name, metadata in st.session_state.processed_pdfs.items():
            with st.sidebar.expander(pdf_name):
                st.write(f"**Pages:** {metadata['num_pages']}")
                if metadata['title'] != 'Unknown':
                    st.write(f"**Title:** {metadata['title']}")
                if metadata['author'] != 'Unknown':
                    st.write(f"**Author:** {metadata['author']}")
                if metadata['subject']:
                    st.write(f"**Subject:** {metadata['subject']}")


def export_chat_history():
    """Export chat history to JSON file"""
    history_json = st.session_state.chat_manager.export_history()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.json"
    
    st.download_button(
        label="💾 Download Chat History",
        data=history_json,
        file_name=filename,
        mime="application/json"
    )


def main():
    """Main application"""
    st.set_page_config(
        page_title="Chat with PDF",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Title and description
    st.title("📚 Chat with PDF")
    st.markdown("Upload PDFs and ask questions using local AI models - completely offline!")
    
    # Sidebar - Settings and Controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # PDF Upload
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with"
        )
        
        # Processing options
        chunk_size = st.slider("Chunk Size (words)", 200, 1000, 500, 50)
        top_k_results = st.slider("Results per Query", 1, 10, 5, 1)
        
        use_ocr = st.checkbox("Enable OCR (slower)", value=False, help="Extract text from images in PDFs")
        
        # Process button
        if uploaded_files:
            if st.button("🔄 Process PDFs", type="primary"):
                with st.spinner("Processing PDFs..."):
                    num_chunks = process_uploaded_pdfs(uploaded_files, chunk_size, use_ocr)
                    st.success(f"✅ Processed {len(uploaded_files)} PDF(s) - {num_chunks} chunks indexed")
        
        st.divider()
        
        # Query type
        st.subheader("Query Settings")
        query_type = st.selectbox(
            "Query Type",
            ["Direct Answer", "Summary", "Keyword Search", "Entity Extraction", "Sentiment Analysis"],
            help="Choose how to process your query"
        )
        
        st.divider()
        
        # Display PDF metadata
        display_pdf_metadata()
        
        st.divider()
        
        # Chat controls
        st.subheader("Chat Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_manager.clear_history()
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset All"):
                st.session_state.embedding_manager.clear()
                st.session_state.chat_manager.clear_history()
                st.session_state.processed_pdfs = {}
                st.rerun()
        
        # Export chat
        if st.session_state.chat_manager.history:
            export_chat_history()
        
        st.divider()
        
        # Statistics
        if st.session_state.processed_pdfs:
            st.subheader("📊 Statistics")
            total_pages = sum(meta['num_pages'] for meta in st.session_state.processed_pdfs.values())
            st.metric("Total Pages", total_pages)
            st.metric("Total Chunks", len(st.session_state.embedding_manager.chunks))
            st.metric("Chat Messages", len(st.session_state.chat_manager.history))
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_manager.history:
            st.info("👋 Upload PDFs and start asking questions!")
        else:
            for message in st.session_state.chat_manager.history:
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(message['content'])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message['content'])
    
    # Query input
    if st.session_state.processed_pdfs:
        query = st.chat_input("Ask a question about your PDFs...")
        
        if query:
            # Display user message
            with st.chat_message("user"):
                st.write(query)
            
            # Store user message
            st.session_state.chat_manager.add_message('user', query)
            
            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = handle_query(query, query_type, top_k_results)
                    st.markdown(response)
            
            # Rerun to update chat display
            st.rerun()
    else:
        st.warning("⚠️ Please upload and process PDFs first!")
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
        <small>Chat with PDF - Powered by Open Source AI | Running 100% Locally</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
