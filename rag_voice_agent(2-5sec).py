# =============================================================================
# OPTIMIZED SETTINGS - src/config/settings.py
# =============================================================================
"""
Ultra-optimized settings for low-latency RAG.
Target: Sub-second response times.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # =========================================================================
    # CRITICAL PERFORMANCE SETTINGS
    # =========================================================================
    
    # Vector Search - Reduce retrieval count
    top_k: int = 3  # Reduced from 5+ (fewer docs = faster)
    
    # Chunking - Smaller chunks = faster processing
    chunk_size: int = 512  # Reduced from 1000+
    chunk_overlap: int = 50  # Reduced from 200+
    
    # LLM - Use fastest models
    max_tokens: int = 256  # Reduced from 1000+ (shorter responses)
    
    # Cache - Essential for speed
    use_cache: bool = True
    cache_ttl: int = 3600
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    # =========================================================================
    # AWS BEDROCK (FASTEST OPTION)
    # =========================================================================
    llm_provider: str = "bedrock"
    
    # Titan Express is THE FASTEST Bedrock model
    bedrock_models: list = [
        "amazon.titan-text-express-v1",  # Fastest, use this!
        "amazon.titan-text-lite-v1",     # Even faster but less capable
    ]
    bedrock_temperature: float = 0.3  # Lower = faster, more deterministic
    
    # Titan Embed V2 - Fastest embeddings
    embedding_provider: str = "bedrock"
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 256  # Reduced from 1024+ (faster search)
    
    # Vector Store - Use S3 for AWS optimization
    vector_store_type: str = "aws_s3"
    aws_s3_vector_bucket: str = "ll-vector-db"
    aws_s3_vector_index: str = "knowledge-base"
    
    # Reranker - DISABLE for speed (reranking adds latency)
    use_reranker: bool = False
    
    # AWS Credentials
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    
    # S3 Data
    aws_s3_bucket: str = "ll-ai-solutions"
    aws_s3_data_prefix: str = "data/"
    
    # =========================================================================
    # BATCH PROCESSING (For ingestion speed)
    # =========================================================================
    batch_size: int = 100  # Process docs in batches
    parallel_processing: bool = True  # Use multiprocessing
    max_workers: int = 4  # CPU cores for parallel processing
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()


# =============================================================================
# OPTIMIZED CACHE - src/cache.py
# =============================================================================
"""
High-performance cache with connection pooling and compression.
"""
import json
import hashlib
import pickle
import zlib
from typing import Optional, Any
import redis
from redis.connection import ConnectionPool

class OptimizedCache:
    def __init__(self):
        self.enabled = settings.use_cache
        if self.enabled:
            # Connection pool for better performance
            self.pool = ConnectionPool(
                host=settings.redis_host,
                port=settings.redis_port,
                decode_responses=False,  # Binary mode for compression
                max_connections=10
            )
            self.client = redis.Redis(connection_pool=self.pool)
        else:
            self.client = None
    
    def _get_key(self, query: str, **kwargs) -> str:
        """Generate cache key."""
        key_data = f"{query}:{json.dumps(kwargs, sort_keys=True)}"
        return f"rag:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def get(self, query: str, **kwargs) -> Optional[Any]:
        """Get cached result with decompression."""
        if not self.enabled:
            return None
        
        try:
            key = self._get_key(query, **kwargs)
            cached = self.client.get(key)
            
            if cached:
                # Decompress and unpickle
                decompressed = zlib.decompress(cached)
                return pickle.loads(decompressed)
        except Exception:
            pass
        
        return None
    
    def set(self, query: str, result: Any, **kwargs):
        """Cache result with compression."""
        if not self.enabled:
            return
        
        try:
            key = self._get_key(query, **kwargs)
            # Pickle and compress
            pickled = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = zlib.compress(pickled, level=6)
            
            self.client.setex(key, settings.cache_ttl, compressed)
        except Exception:
            pass


# =============================================================================
# OPTIMIZED EMBEDDINGS - src/core/embeddings.py
# =============================================================================
"""
Optimized embeddings with batching and caching.
"""
import sys
from typing import List
import boto3
import json
from functools import lru_cache

class FastEmbeddings:
    def __init__(self):
        self.bedrock = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        self.model_id = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
    
    @lru_cache(maxsize=1000)  # Cache embeddings
    def embed_query(self, text: str) -> List[float]:
        """Embed single query with caching."""
        return self._embed_texts([text])[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents in batches."""
        batch_size = 25  # Bedrock batch limit
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._embed_texts(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Call Bedrock embedding API."""
        body = json.dumps({
            "inputText": texts[0] if len(texts) == 1 else texts,
            "dimensions": self.dimensions,
            "normalize": True
        })
        
        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=body
        )
        
        result = json.loads(response['body'].read())
        
        if len(texts) == 1:
            return [result['embedding']]
        return result['embeddings']


# =============================================================================
# OPTIMIZED LLM - src/core/llm_provider.py
# =============================================================================
"""
Ultra-fast LLM with streaming disabled and minimal tokens.
"""
import sys
import json
import boto3
from typing import List, Dict

class FastLLM:
    def __init__(self):
        self.bedrock = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        # Use fastest model
        self.model_id = "amazon.titan-text-express-v1"
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response with minimal latency."""
        # Format prompt
        prompt = self._format_messages(messages)
        
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "temperature": 0.3,  # Lower = faster
                "maxTokenCount": settings.max_tokens,
                "topP": 0.9,
                "stopSequences": []
            }
        })
        
        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=body
        )
        
        result = json.loads(response['body'].read())
        return result['results'][0]['outputText'].strip()
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages for Titan."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"{content}\n")
            elif role == "user":
                parts.append(f"Question: {content}\n")
        parts.append("Answer:")
        return "\n".join(parts)


# =============================================================================
# OPTIMIZED VECTOR STORE - src/core/vectorstore.py
# =============================================================================
"""
Optimized vector store with minimal overhead.
"""
import sys
from typing import List
from langchain_core.documents import Document
from langchain_aws.vectorstores import AmazonS3Vectors

class FastVectorStore:
    def __init__(self, embeddings: FastEmbeddings):
        self.embeddings = embeddings
        self.vector_store = AmazonS3Vectors(
            vector_bucket_name=settings.aws_s3_vector_bucket,
            index_name=settings.aws_s3_vector_index,
            embedding=embeddings,
            create_index_if_not_exist=True
        )
    
    def add_documents(self, documents: List[Document]):
        """Add documents in optimized batches."""
        batch_size = settings.batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vector_store.add_documents(batch)
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Fast similarity search with minimal k."""
        return self.vector_store.similarity_search(query, k=k)


# =============================================================================
# OPTIMIZED INGESTION - src/core/ingestion.py
# =============================================================================
"""
Parallel document processing for faster ingestion.
"""
import sys
from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class FastIngestion:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
            length_function=len
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load single document."""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        try:
            if ext == ".pdf":
                loader = PyMuPDFLoader(str(file_path))
            elif ext == ".txt":
                loader = TextLoader(str(file_path))
            else:
                return []
            
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = str(file_path)
            return docs
        except Exception:
            return []
    
    def process_directory(self, directory: str) -> List[Document]:
        """Process directory in parallel."""
        dir_path = Path(directory)
        files = list(dir_path.rglob('*'))
        files = [f for f in files if f.is_file() and f.suffix.lower() in ['.pdf', '.txt']]
        
        all_docs = []
        
        # Parallel loading
        with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
            futures = {executor.submit(self.load_document, str(f)): f for f in files}
            
            for future in as_completed(futures):
                docs = future.result()
                all_docs.extend(docs)
        
        # Chunk documents
        chunks = self.text_splitter.split_documents(all_docs)
        return chunks


# =============================================================================
# OPTIMIZED MAIN - main.py
# =============================================================================
"""
Ultra-fast RAG query with caching and optimizations.
"""
import time
from typing import Dict

# Initialize cache globally
cache = OptimizedCache()

def fast_query(question: str, vector_store: FastVectorStore, llm: FastLLM) -> Dict:
    """Execute query with sub-second latency."""
    start = time.perf_counter()
    
    # Check cache first
    cached = cache.get(question, top_k=settings.top_k)
    if cached:
        cached['from_cache'] = True
        cached['total_time'] = time.perf_counter() - start
        return cached
    
    # Retrieve documents (fast: reduced k=3)
    retrieval_start = time.perf_counter()
    docs = vector_store.similarity_search(question, k=settings.top_k)
    retrieval_time = time.perf_counter() - retrieval_start
    
    if not docs:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "retrieval_time": retrieval_time,
            "llm_time": 0,
            "total_time": time.perf_counter() - start
        }
    
    # Build context (minimal)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Generate answer (fast: short max_tokens)
    llm_start = time.perf_counter()
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Provide concise, accurate answers based on the context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a brief, direct answer."
        }
    ]
    
    answer = llm.generate(messages)
    llm_time = time.perf_counter() - llm_start
    
    # Prepare result
    result = {
        "answer": answer,
        "sources": [doc.metadata.get("source", "Unknown") for doc in docs],
        "retrieval_time": retrieval_time,
        "llm_time": llm_time,
        "total_time": time.perf_counter() - start,
        "from_cache": False
    }
    
    # Cache result
    cache.set(question, result, top_k=settings.top_k)
    
    return result


def benchmark_query(question: str):
    """Benchmark single query."""
    print("="*80)
    print(f"QUERY: {question}")
    print("="*80)
    
    # Initialize components
    embeddings = FastEmbeddings()
    vector_store = FastVectorStore(embeddings)
    llm = FastLLM()
    
    # Execute query
    result = fast_query(question, vector_store, llm)
    
    # Display results
    print(f"\n📚 Answer:")
    print(f"   {result['answer']}")
    
    print(f"\n📄 Sources: {len(result['sources'])}")
    for i, source in enumerate(result['sources'], 1):
        print(f"   {i}. {source}")
    
    print(f"\n⏱️  Performance:")
    print(f"   Retrieval: {result['retrieval_time']*1000:.2f} ms")
    print(f"   LLM:       {result['llm_time']*1000:.2f} ms")
    print(f"   Total:     {result['total_time']*1000:.2f} ms")
    print(f"   Cached:    {'✅ Yes' if result['from_cache'] else '❌ No'}")


def main():
    """Main execution."""
    print("="*80)
    print("ULTRA-FAST RAG SYSTEM")
    print("="*80)
    
    # Test queries
    test_queries = [
        "What is the main topic?",
        "Summarize the key points.",
        "What are the conclusions?"
    ]
    
    for query in test_queries:
        benchmark_query(query)
        print("\n")


if __name__ == "__main__":
    main()