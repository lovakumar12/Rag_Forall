# =============================================================================
# ULTRA-FAST RAG SYSTEM - TARGET: 20ms
# =============================================================================
"""
Extreme performance RAG using:
- In-memory vector store (FAISS with HNSW)
- Cached embeddings
- Groq (fastest LLM: 500+ tokens/sec)
- Sentence Transformers (local, no network)
- Aggressive caching
- Pre-computed contexts
"""


import os
import sys
import time
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from functools import lru_cache

# Fast local embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

# FAISS for ultra-fast vector search
import faiss

# Groq for fastest LLM (500+ tokens/sec)
from groq import Groq

# Document processing
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


print("DEBUG GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class UltraFastConfig:
    """Ultra-optimized configuration for 20ms target."""
    
    # Embedding model (fastest local model)
    embedding_model: str = "all-MiniLM-L6-v2"  # 384 dims, 14ms on CPU
    
    # LLM (Groq is fastest: 500+ tokens/sec)
    groq_api_key: str = os.getenv("GROQ_API_KEY","")
    groq_model: str = "llama-3.1-8b-instant"  # Fastest Groq model
    
    # Vector search (aggressive optimization)
    top_k: int = 1  # Only retrieve 1 document for speed
    max_docs_in_db: int = 1000  # Limit corpus size for speed
    
    # Chunking (tiny chunks for speed)
    chunk_size: int = 256  # Very small chunks
    chunk_overlap: int = 20
    
    # LLM generation (minimal tokens)
    max_tokens: int = 50  # Very short answers
    temperature: float = 0.1  # Deterministic, faster
    
    # Cache settings
    cache_dir: str = ".cache"
    use_cache: bool = True
    
    # FAISS index type (HNSW is fastest for search)
    faiss_index_type: str = "HNSW"
    faiss_ef_search: int = 16  # Lower = faster search
    faiss_m: int = 16  # HNSW parameter


config = UltraFastConfig()


# =============================================================================
# ULTRA-FAST EMBEDDINGS (Local, No Network)
# =============================================================================

class LightningEmbeddings:
    """Lightning-fast local embeddings with caching."""
    
    def __init__(self):
        print("⚡ Loading embedding model (one-time setup)...")
        self.model = SentenceTransformer(config.embedding_model)
        self.model.max_seq_length = 128  # Truncate for speed
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # In-memory cache for embeddings
        self._cache = {}
        
        print(f"✅ Embedding model loaded: {self.dimension} dimensions")
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed query with caching."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        embedding = self.model.encode(text, normalize_embeddings=True)
        self._cache[cache_key] = embedding
        return embedding
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Batch embed documents (fast)."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )
        return embeddings


# =============================================================================
# ULTRA-FAST VECTOR STORE (In-Memory FAISS)
# =============================================================================

class LightningVectorStore:
    """In-memory FAISS with HNSW for sub-ms search."""
    
    def __init__(self, embeddings: LightningEmbeddings):
        self.embeddings = embeddings
        self.dimension = embeddings.dimension
        self.index = None
        self.documents = []
        self.metadata = []
        self.index_path = Path(config.cache_dir) / "faiss_index"
        
        # Try to load existing index
        if self.index_path.exists():
            self._load_index()
    
    def _create_index(self):
        """Create HNSW index (fastest for search)."""
        # HNSW index for maximum speed
        self.index = faiss.IndexHNSWFlat(self.dimension, config.faiss_m)
        self.index.hnsw.efConstruction = 40
        self.index.hnsw.efSearch = config.faiss_ef_search
        print(f"✅ Created HNSW index: {self.dimension}D")
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to index."""
        if self.index is None:
            self._create_index()
        
        # Limit corpus size for speed
        if len(texts) > config.max_docs_in_db:
            print(f"⚠️  Limiting to {config.max_docs_in_db} documents for speed")
            texts = texts[:config.max_docs_in_db]
            metadata = metadata[:config.max_docs_in_db] if metadata else None
        
        print(f"⚡ Embedding {len(texts)} documents...")
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add to FAISS
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])
        
        print(f"✅ Added {len(texts)} documents to index")
        
        # Save index
        self._save_index()
    
    def search(self, query: str, k: int = 1) -> List[Dict[str, Any]]:
        """Ultra-fast search (target: <5ms)."""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search FAISS (sub-millisecond)
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(1 - distance)  # Convert distance to similarity
                })
        
        return results
    
    def _save_index(self):
        """Save index to disk."""
        self.index_path.parent.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path) + ".faiss")
        
        # Save documents and metadata
        with open(str(self.index_path) + ".pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        
        print(f"💾 Index saved to {self.index_path}")
    
    def _load_index(self):
        """Load index from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path) + ".faiss")
            
            # Load documents and metadata
            with open(str(self.index_path) + ".pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            
            print(f"✅ Loaded index with {len(self.documents)} documents")
        except Exception as e:
            print(f"⚠️  Could not load index: {e}")
            self.index = None


# =============================================================================
# ULTRA-FAST LLM (Groq: 500+ tokens/sec)
# =============================================================================

class LightningLLM:
    """Groq LLM for maximum speed (500+ tokens/sec)."""
    
    def __init__(self):
        if not config.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Get one at: https://console.groq.com/keys\n"
                "Set it: export GROQ_API_KEY='your-key'"
            )
        
        self.client = Groq(api_key=config.groq_api_key)
        self.model = config.groq_model
        print(f"✅ Groq initialized: {self.model}")
    
    def generate(self, context: str, question: str) -> str:
        """Generate answer (Groq is 10x faster than others)."""
        
        # Ultra-concise prompt for speed
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer (be brief):"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"


# =============================================================================
# DOCUMENT INGESTION
# =============================================================================

class FastIngestion:
    """Fast document processing."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
            length_function=len
        )
    
    def load_and_process(self, directory: str) -> tuple:
        """Load and chunk documents."""
        print(f"\n📂 Loading documents from: {directory}")
        
        all_docs = []
        dir_path = Path(directory)
        
        # Load documents
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                
                try:
                    if ext == '.txt':
                        loader = TextLoader(str(file_path))
                        docs = loader.load()
                        all_docs.extend(docs)
                    elif ext == '.pdf':
                        loader = PyMuPDFLoader(str(file_path))
                        docs = loader.load()
                        all_docs.extend(docs)
                except Exception as e:
                    print(f"⚠️  Skipped {file_path.name}: {e}")
        
        if not all_docs:
            raise ValueError(f"No documents loaded from {directory}")
        
        print(f"✅ Loaded {len(all_docs)} documents")
        
        # Chunk documents
        print(f"⚡ Chunking documents...")
        chunks = self.text_splitter.split_documents(all_docs)
        
        # Extract texts and metadata
        texts = [chunk.page_content for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        
        print(f"✅ Created {len(chunks)} chunks")
        
        return texts, metadata


# =============================================================================
# ULTRA-FAST RAG SYSTEM
# =============================================================================

class LightningRAG:
    """Complete RAG system optimized for 20ms queries."""
    
    def __init__(self):
        print("\n" + "="*80)
        print("⚡ LIGHTNING RAG - TARGET: 20ms")
        print("="*80)
        
        self.embeddings = LightningEmbeddings()
        self.vector_store = LightningVectorStore(self.embeddings)
        self.llm = LightningLLM()
        
        # Query cache (in-memory for speed)
        self._query_cache = {}
        
        print("\n✅ Lightning RAG initialized")
    
    def ingest_documents(self, directory: str):
        """Ingest documents from directory."""
        ingestion = FastIngestion()
        texts, metadata = ingestion.load_and_process(directory)
        self.vector_store.add_documents(texts, metadata)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Execute query with timing."""
        start = time.perf_counter()
        
        # Check cache
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if config.use_cache and cache_key in self._query_cache:
            result = self._query_cache[cache_key].copy()
            result['from_cache'] = True
            result['total_time_ms'] = (time.perf_counter() - start) * 1000
            return result
        
        # Retrieve documents
        retrieval_start = time.perf_counter()
        docs = self.vector_store.search(question, k=config.top_k)
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000
        
        if not docs:
            return {
                'answer': 'No relevant documents found.',
                'sources': [],
                'retrieval_time_ms': retrieval_time,
                'llm_time_ms': 0,
                'total_time_ms': (time.perf_counter() - start) * 1000,
                'from_cache': False
            }
        
        # Build context (minimal)
        context = docs[0]['text']  # Only use top 1 doc for speed
        
        # Generate answer
        llm_start = time.perf_counter()
        answer = self.llm.generate(context, question)
        llm_time = (time.perf_counter() - llm_start) * 1000
        
        # Prepare result
        total_time = (time.perf_counter() - start) * 1000
        
        result = {
            'answer': answer,
            'sources': [doc['metadata'].get('source', 'Unknown') for doc in docs],
            'retrieval_time_ms': retrieval_time,
            'llm_time_ms': llm_time,
            'total_time_ms': total_time,
            'from_cache': False
        }
        
        # Cache result
        self._query_cache[cache_key] = result
        
        return result


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def format_time(ms: float) -> str:
    """Format milliseconds nicely."""
    if ms < 1:
        return f"{ms*1000:.0f}μs"
    return f"{ms:.1f}ms"


def print_separator(title: str = ""):
    """Print visual separator."""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)


def benchmark_queries(rag: LightningRAG):
    """Benchmark multiple queries."""
    test_queries = [
        "What is the main topic?",
        "Summarize the key points.",
        "What are the conclusions?",
        "Explain the methodology.",
        "What are the results?"
    ]
    
    print_separator("BENCHMARK RESULTS")
    
    all_times = []
    cache_times = []
    
    for i, question in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {question}")
        
        # First run (no cache)
        result = rag.query(question)
        print(f"   ⏱️  Total: {format_time(result['total_time_ms'])} "
              f"(Retrieval: {format_time(result['retrieval_time_ms'])}, "
              f"LLM: {format_time(result['llm_time_ms'])})")
        print(f"   💬 Answer: {result['answer'][:100]}...")
        
        all_times.append(result['total_time_ms'])
        
        # Second run (cached)
        cached_result = rag.query(question)
        print(f"   ⚡ Cached: {format_time(cached_result['total_time_ms'])}")
        cache_times.append(cached_result['total_time_ms'])
    
    # Statistics
    print_separator("STATISTICS")
    print(f"\n📊 Performance Metrics:")
    print(f"   First-time queries:")
    print(f"      Average: {format_time(np.mean(all_times))}")
    print(f"      Min: {format_time(np.min(all_times))}")
    print(f"      Max: {format_time(np.max(all_times))}")
    
    print(f"\n   Cached queries:")
    print(f"      Average: {format_time(np.mean(cache_times))}")
    print(f"      Min: {format_time(np.min(cache_times))}")
    
    # Check if target met
    avg_time = np.mean(all_times)
    target = 20
    
    print(f"\n🎯 Target: {target}ms")
    if avg_time <= target:
        print(f"   ✅ ACHIEVED! Average: {format_time(avg_time)}")
    else:
        print(f"   ⚠️  Close! Average: {format_time(avg_time)} "
              f"({format_time(avg_time - target)} over target)")
    
    print(f"\n💡 Note: Cached queries averaged {format_time(np.mean(cache_times))}")


def interactive_mode(rag: LightningRAG):
    """Interactive Q&A."""
    print_separator("INTERACTIVE MODE")
    print("\nCommands: 'exit' or 'quit' to exit\n")
    
    while True:
        question = input("❓ Your question: ").strip()
        
        if question.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break
        
        if question:
            result = rag.query(question)
            
            print(f"\n💬 Answer: {result['answer']}")
            print(f"⏱️  Time: {format_time(result['total_time_ms'])} "
                  f"{'(cached)' if result['from_cache'] else ''}")
            print(f"📚 Sources: {', '.join(result['sources'])}\n")
