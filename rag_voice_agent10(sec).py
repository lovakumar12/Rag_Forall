#==============================================
# Redis Cache Implementation--->src/src/cache.py
#==============================================

import json
import hashlib
from typing import Optional, Any
import redis
from src.config.settings import settings

class Cache:
    def __init__(self):
        self.enabled = settings.use_cache
        if self.enabled:
            self.client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                decode_responses=True
            )
        else:
            self.client = None
    
    def _get_key(self, query: str, **kwargs) -> str:
        """Generate cache key from query and parameters."""
        key_data = f"{query}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, **kwargs) -> Optional[Any]:
        """Get cached result."""
        if not self.enabled:
            return None
        
        key = self._get_key(query, **kwargs)
        cached = self.client.get(key)
        
        if cached:
            return json.loads(cached)
        return None
    
    def set(self, query: str, result: Any, **kwargs):
        """Cache a result."""
        if not self.enabled:
            return
        
        key = self._get_key(query, **kwargs)
        self.client.setex(
            key,
            settings.cache_ttl,
            json.dumps(result)
        )
    
    def clear(self):
        """Clear all cache."""
        if self.enabled:
            self.client.flushdb()



# ============================================================================
# FILE: src/core/ingestion.py 
# ============================================================================
"""
Document ingestion system for the RAG pipeline.
Supports:
- Multiple document formats (PDF, DOCX, TXT, CSV, HTML, etc.)
- S3 document loading from ll-ai-solutions/data/
- Local directory loading
- Automatic chunking with optimal parameters
- Keyword extraction for better retrieval

Optimized for low latency and high throughput.
"""
import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Type, Optional
from langchain_core.documents import Document

from langchain_community.document_loaders import (
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.config.settings import settings
from src.core.s3_loader import S3DocumentLoader

logger = get_logger(__name__)

# Mapping of file extensions to document loaders
LOADER_MAPPING: Dict[str, Type] = {
    ".csv": CSVLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".enex": EverNoteLoader,
    ".epub": UnstructuredEPubLoader,
    ".html": UnstructuredHTMLLoader,
    ".md": UnstructuredMarkdownLoader,
    ".odt": UnstructuredODTLoader,
    ".pdf": PyMuPDFLoader,  # Fast PDF loader
    ".ppt": UnstructuredPowerPointLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".txt": TextLoader,
    ".eml": UnstructuredEmailLoader,
}


class DocumentIngestion:
    """
    Document ingestion and processing system.
    Handles loading, chunking, and preparation of documents for embedding.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        load_from_s3: bool = False
    ):
        """
        Initialize document ingestion system.
        
        Args:
            chunk_size: Size of text chunks (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
            load_from_s3: Whether to load documents from S3
        """
        try:
            logger.info("Initializing DocumentIngestion system")
            
            self.chunk_size = chunk_size or settings.chunk_size
            self.chunk_overlap = chunk_overlap or settings.chunk_overlap
            self.load_from_s3 = load_from_s3
            
            # Initialize text splitter for chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
            
            logger.info(f"DocumentIngestion initialized - chunk_size: {self.chunk_size}, overlap: {self.chunk_overlap}")
            
            # Initialize S3 loader if needed
            if self.load_from_s3:
                self.s3_loader = S3DocumentLoader()
                logger.info("S3DocumentLoader initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize DocumentIngestion: {str(e)}")
            raise CustomException(e, sys)
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document based on its file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            logger.info(f"Loading document: {path.name} (type: {extension})")
            
            # Get appropriate loader
            loader_class = LOADER_MAPPING.get(extension)
            if not loader_class:
                logger.warning(f"Unsupported file type: {extension} for file: {path.name}")
                return []
            
            # Load document
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata['source'] = str(file_path)
                doc.metadata['filename'] = path.name
                doc.metadata['file_type'] = extension
            
            logger.info(f"Successfully loaded {len(documents)} document(s) from {path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of all loaded Document objects
        """
        try:
            dir_path = Path(directory_path)
            
            if not dir_path.exists() or not dir_path.is_dir():
                logger.error(f"Invalid directory path: {directory_path}")
                raise ValueError(f"Directory does not exist: {directory_path}")
            
            logger.info(f"Loading documents from directory: {directory_path}")
            
            all_documents = []
            file_count = 0
            
            # Process each file in directory
            for file_path in dir_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in LOADER_MAPPING:
                    docs = self.load_document(str(file_path))
                    all_documents.extend(docs)
                    if docs:
                        file_count += 1
            
            logger.info(f"Loaded {len(all_documents)} documents from {file_count} files")
            return all_documents
            
        except Exception as e:
            logger.error(f"Error loading directory {directory_path}: {str(e)}")
            raise CustomException(e, sys)
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for embedding.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        try:
            if not documents:
                logger.warning("No documents to chunk")
                return []
            
            logger.info(f"Chunking {len(documents)} documents...")
            
            chunks = self.text_splitter.split_documents(documents)
            
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            raise CustomException(e, sys)
    
    def process_documents(
        self,
        source_path: Optional[str] = None,
        from_s3: bool = None
    ) -> List[Document]:
        """
        Main method to process documents from various sources.
        
        Args:
            source_path: Local directory path (optional if loading from S3)
            from_s3: Override self.load_from_s3 flag
            
        Returns:
            List of processed and chunked Document objects
        """
        try:
            use_s3 = from_s3 if from_s3 is not None else self.load_from_s3
            temp_dir = None
            
            # Load from S3 if specified
            if use_s3:
                logger.info("Loading documents from S3...")
                temp_dir = self.s3_loader.load_documents_from_s3()
                
                if not temp_dir:
                    logger.error("Failed to load documents from S3")
                    raise ValueError("No documents loaded from S3")
                
                source_path = temp_dir
            
            # Validate source path
            if not source_path:
                logger.error("No source path provided and S3 loading not enabled")
                raise ValueError("Must provide source_path or enable S3 loading")
            
            # Load documents
            logger.info(f"Processing documents from: {source_path}")
            documents = self.load_directory(source_path)
            
            if not documents:
                logger.warning(f"No documents loaded from {source_path}")
                return []
            
            # Chunk documents
            chunks = self.chunk_documents(documents)
            
            # Cleanup temp directory if used
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            
            logger.info(f"Successfully processed {len(chunks)} document chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            # Cleanup temp directory on error
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise CustomException(e, sys)
        
        
# ============================================================================
# FILE: src/core/llm_provider.py (MULTI-PROVIDER)
# ============================================================================
"""
Multi-provider LLM system for the RAG pipeline.
Priority Order: AWS Bedrock (Titan) → OpenAI → Groq
Automatically falls back to next provider if one fails.

AWS Bedrock Models (Priority 1):
1. amazon.titan-text-express-v1 (fastest, cheapest)
2. amazon.titan-text-lite-v1 (ultra fast)
3. anthropic.claude-3-haiku (fast, good quality)
4. anthropic.claude-3-5-sonnet (best quality)

OpenAI Models (Priority 2):
- gpt-3.5-turbo (fast, cost-effective)
- gpt-4 (high quality)

Groq Models (Priority 3):
- mixtral-8x7b-32768 (very fast)
- llama2-70b-4096 (good quality)
"""
import sys
import json
from typing import List, Dict, Any, Iterator, Optional
import boto3

# OpenAI import with error handling
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Groq import with error handling
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.config.settings import settings

logger = get_logger(__name__)


class MultiProviderLLM:
    """
    Multi-provider LLM with automatic fallback.
    Priority: Bedrock (Titan) → OpenAI → Groq
    """
    
    def __init__(self):
        """Initialize LLM providers based on configuration."""
        try:
            logger.info("Initializing MultiProviderLLM")
            self.provider = None
            self.llm = None
            self.bedrock_client = None
            self.current_model = None
            
            # Initialize LLM with fallback logic
            self._initialize_llm()
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise CustomException(e, sys)
    
    def _initialize_llm(self):
        """Initialize LLM provider with fallback logic."""
        
        # Priority 1: AWS Bedrock (with Titan priority)
        if settings.llm_provider == "bedrock" or self.llm is None:
            if self._init_bedrock():
                return
        
        # Priority 2: OpenAI
        if settings.llm_provider == "openai" or self.llm is None:
            if self._init_openai():
                return
        
        # Priority 3: Groq
        if settings.llm_provider == "groq" or self.llm is None:
            if self._init_groq():
                return
        
        # If all failed
        logger.error("❌ All LLM providers failed")
        raise Exception("No LLM provider could be initialized. Please configure at least one provider.")
    
    def _init_bedrock(self) -> bool:
        """
        Initialize AWS Bedrock LLM with model fallback.
        Tries models in priority order from settings.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Attempting to initialize AWS Bedrock LLM...")
            
            # Check credentials
            if not settings.aws_access_key_id or not settings.aws_secret_access_key:
                logger.warning("AWS credentials not configured. Skipping Bedrock.")
                logger.warning("Please configure: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY in .env")
                return False
            
            # Initialize Bedrock client
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region
            )
            
            # Try models in priority order
            for model_id in settings.bedrock_models:
                try:
                    logger.info(f"Testing Bedrock model: {model_id}")
                    
                    # Test the model with a simple query
                    test_response = self._invoke_bedrock_model(
                        model_id=model_id,
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=10
                    )
                    
                    if test_response:
                        self.provider = "bedrock"
                        self.current_model = model_id
                        logger.info(f"✅ Bedrock initialized successfully with model: {model_id}")
                        return True
                        
                except Exception as e:
                    logger.warning(f"Model {model_id} failed: {str(e)}")
                    continue
            
            logger.warning("All Bedrock models failed")
            return False
            
        except Exception as e:
            logger.warning(f"Bedrock initialization failed: {str(e)}")
            return False
    
    def _init_openai(self) -> bool:
        """
        Initialize OpenAI LLM.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI package not available. Install with: pip install langchain-openai")
                return False
            
            if not settings.openai_api_key:
                logger.warning("OpenAI API key not configured. Skipping OpenAI.")
                logger.warning("Please configure: OPENAI_API_KEY in .env")
                return False
            
            logger.info("Attempting to initialize OpenAI LLM...")
            
            self.llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=settings.openai_temperature,
                max_tokens=settings.max_tokens,
                openai_api_key=settings.openai_api_key
            )
            
            # Test the model
            test_response = self.llm.invoke("Hi")
            if test_response:
                self.provider = "openai"
                self.current_model = settings.openai_model
                logger.info(f"✅ OpenAI initialized successfully with model: {settings.openai_model}")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"OpenAI initialization failed: {str(e)}")
            return False
    
    def _init_groq(self) -> bool:
        """
        Initialize Groq LLM.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not GROQ_AVAILABLE:
                logger.warning("Groq package not available. Install with: pip install langchain-groq")
                return False
            
            if not settings.groq_api_key:
                logger.warning("Groq API key not configured. Skipping Groq.")
                logger.warning("Please configure: GROQ_API_KEY in .env")
                return False
            
            logger.info("Attempting to initialize Groq LLM...")
            
            self.llm = ChatGroq(
                model=settings.groq_model,
                temperature=settings.groq_temperature,
                max_tokens=settings.max_tokens,
                groq_api_key=settings.groq_api_key
            )
            
            # Test the model
            test_response = self.llm.invoke("Hi")
            if test_response:
                self.provider = "groq"
                self.current_model = settings.groq_model
                logger.info(f"✅ Groq initialized successfully with model: {settings.groq_model}")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Groq initialization failed: {str(e)}")
            return False
    
    def _invoke_bedrock_model(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """
        Invoke a Bedrock model.
        
        Args:
            model_id: Bedrock model ID
            messages: List of message dicts with role and content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        try:
            max_tokens = max_tokens or settings.max_tokens
            temperature = temperature or settings.bedrock_temperature
            
            # Format request based on model family
            if "anthropic.claude" in model_id:
                # Claude format
                system_message = ""
                conversation_messages = []
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        conversation_messages.append(msg)
                
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": conversation_messages
                }
                
                if system_message:
                    body["system"] = system_message
                
            elif "amazon.titan" in model_id:
                # Titan format
                prompt = self._format_messages_to_prompt(messages)
                body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "temperature": temperature,
                        "maxTokenCount": max_tokens,
                        "topP": 0.9
                    }
                }
            
            else:
                raise ValueError(f"Unsupported model: {model_id}")
            
            # Invoke model
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            if "anthropic.claude" in model_id:
                return response_body['content'][0]['text']
            elif "amazon.titan" in model_id:
                return response_body['results'][0]['outputText']
            
        except Exception as e:
            logger.error(f"Error invoking Bedrock model {model_id}: {str(e)}")
            raise CustomException(e, sys)
    
    def _format_messages_to_prompt(self, messages: List[Dict]) -> str:
        """Format messages into a single prompt string for Titan."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            messages: List of message dicts with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            logger.info(f"Generating response using {self.provider} ({self.current_model})")
            
            if self.provider == "bedrock":
                response = self._invoke_bedrock_model(
                    model_id=self.current_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            elif self.provider in ["openai", "groq"]:
                # Convert messages to LangChain format
                from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
                
                lc_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        lc_messages.append(SystemMessage(content=msg["content"]))
                    elif msg["role"] == "user":
                        lc_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        lc_messages.append(AIMessage(content=msg["content"]))
                
                response = self.llm.invoke(lc_messages)
                response = response.content
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            logger.info(f"Successfully generated response ({len(response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise CustomException(e, sys)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current LLM."""
        return {
            "provider": self.provider,
            "model": self.current_model,
            "available_models": {
                "bedrock": settings.bedrock_models if self.bedrock_client else [],
                "openai": [settings.openai_model] if settings.openai_api_key else [],
                "groq": [settings.groq_model] if settings.groq_api_key else []
            }
        }


# ============================================================================
# FILE: src/core/reranker.py
# ============================================================================
# ============================================================================
# FILE: src/core/reranker.py
# ============================================================================

from typing import List, Dict, Any
import cohere
from src.config.settings import settings

class Reranker:
    def __init__(self, model: str = None):
        self.model = model or settings.reranker_model
        self.client = cohere.Client(settings.cohere_api_key) if settings.cohere_api_key else None
        self.enabled = settings.use_reranker and self.client is not None
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents using Cohere's reranker."""
        if not self.enabled or not documents:
            return documents[:top_n] if top_n else documents
        
        top_n = top_n or settings.rerank_top_n
        texts = [doc['text'] for doc in documents]
        
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=texts,
            top_n=top_n
        )
        
        reranked_docs = []
        for result in response.results:
            doc = documents[result.index].copy()
            doc['rerank_score'] = result.relevance_score
            reranked_docs.append(doc)
        
        return reranked_docs
    
    
# ============================================================================
# FILE: src/core/s3_loader.py
# ============================================================================


import os
import tempfile
from pathlib import Path
import boto3
import sys


from src.config.settings import settings
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


class S3DocumentLoader:
    """
    Downloads documents from S3 into a temporary local directory
    and returns the directory path.
    """

    def __init__(self):
        try:
            self.bucket_name = settings.aws_s3_bucket
            self.prefix = settings.aws_s3_data_prefix  # e.g. "data/"
            
            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region,
            )

            logger.info(f"S3DocumentLoader initialized for bucket: {self.bucket_name}")

        except Exception as e:
            raise CustomException(e,sys)
        

    def load_documents_from_s3(self) -> str:
        """
        Downloads all files from S3 prefix into a temp directory.
        
        Returns:
            str: Local temp directory path
        """
        try:
            temp_dir = tempfile.mkdtemp(prefix="s3_docs_")
            logger.info(f"Downloading S3 documents to temp dir: {temp_dir}")

            paginator = self.s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=self.prefix
            )

            file_count = 0

            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]

                    if key.endswith("/"):
                        continue  # skip folders

                    local_path = Path(temp_dir) / Path(key).name
                    self.s3.download_file(self.bucket_name, key, str(local_path))
                    file_count += 1

            if file_count == 0:
                logger.warning("No files downloaded from S3")

            logger.info(f"Downloaded {file_count} files from S3")
            return temp_dir

        except Exception as e:
            logger.error("Failed to download documents from S3")
            raise CustomException(e, sys)



# ============================================================================
# FILE: src/core/vectorstore.py (MULTI-PROVIDER)
# ============================================================================
"""
Multi-provider vector store system for the RAG pipeline.
Priority Order: AWS S3 Vectors → Milvus → FAISS
Automatically falls back to next provider if one fails.

AWS S3 (Priority 1):
- Uses AWS credits efficiently
- Serverless, no infrastructure management
- Bucket: ll-vector-db, Index: knowledge-base

Milvus (Priority 2):
- High performance for production
- Requires separate Milvus server

FAISS (Priority 3):
- Local, no external dependencies
- Good for development and testing
"""
import sys
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_aws.vectorstores import AmazonS3Vectors

# Milvus import with error handling
try:
    from langchain_milvus import Milvus
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("Milvus not installed. Install with: pip install langchain-milvus pymilvus")

from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.config.settings import settings
from src.core.embeddings import MultiProviderEmbeddings

logger = get_logger(__name__)


class MultiVectorStore:
    """
    Multi-provider vector store with automatic fallback.
    Priority: AWS S3 → Milvus → FAISS
    """
    
    def __init__(self, embeddings: MultiProviderEmbeddings = None):
        """
        Initialize vector store with embeddings.
        
        Args:
            embeddings: MultiProviderEmbeddings instance
        """
        try:
            logger.info("Initializing MultiVectorStore")
            
            self.embeddings = embeddings or MultiProviderEmbeddings()
            self.vector_store = None
            self.store_type = None
            
            # Initialize vector store based on priority
            self._initialize_vector_store()
            
        except Exception as e:
            logger.error(f"Failed to initialize MultiVectorStore: {str(e)}")
            raise CustomException(e, sys)
    
    def _initialize_vector_store(self):
        """Initialize vector store with fallback logic."""
        
        # Priority 1: AWS S3 Vectors (best with AWS credits)
        if settings.vector_store_type == "aws_s3" or self.vector_store is None:
            if self._init_aws_s3():
                return
        
        # Priority 2: Milvus (high performance)
        if settings.vector_store_type == "milvus" or self.vector_store is None:
            if self._init_milvus():
                return
        
        # Priority 3: FAISS (fallback, always works)
        if self._init_faiss():
            return
        
        # If all failed
        logger.error("❌ All vector store providers failed")
        raise Exception("No vector store could be initialized. Please configure at least one provider.")
    
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the active vector store.
        """
        try:
            if not self.vector_store:
                raise RuntimeError("Vector store is not initialized")

            if not hasattr(self.vector_store, "add_documents"):
                raise NotImplementedError(
                    f"{type(self.vector_store).__name__} does not support add_documents()"
                )

            return self.vector_store.add_documents(documents)

        except Exception as e:
            raise CustomException(e, sys)

    def similarity_search(self, query: str, k: int = 5):
        """
        Perform similarity search using the active vector store.
        """
        try:
            return self.vector_store.similarity_search(query, k)
        except Exception as e:
            raise CustomException(e, sys)

    def as_retriever(self, **kwargs):
        """
        Return a retriever from the active vector store.
        """
        try:
            return self.vector_store.as_retriever(**kwargs)
        except Exception as e:
            raise CustomException(e, sys)

    
    
    def _init_aws_s3(self) -> bool:
        """
        Initialize AWS S3 vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Attempting to initialize AWS S3 vector store...")
            
            # Check if AWS credentials are available
            if not settings.aws_access_key_id or not settings.aws_secret_access_key:
                logger.warning("AWS credentials not configured. Skipping S3 vector store.")
                logger.warning("Please configure: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY in .env")
                return False
            
            # Try to create/connect to S3 vector store
            self.vector_store = AmazonS3Vectors(
                vector_bucket_name=settings.aws_s3_vector_bucket,
                index_name=settings.aws_s3_vector_index,
                embedding=self.embeddings.get_langchain_embeddings(),
                create_index_if_not_exist=True
            )
            
            self.store_type = "aws_s3"
            logger.info(f"✅ AWS S3 vector store initialized successfully")
            logger.info(f"   Bucket: {settings.aws_s3_vector_bucket}, Index: {settings.aws_s3_vector_index}")
            return True
            
        except Exception as e:
            logger.warning(f"AWS S3 vector store initialization failed: {str(e)}")
            logger.warning("Falling back to next provider...")
            return False
    
    def _init_milvus(self) -> bool:
        """
        Initialize Milvus vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not MILVUS_AVAILABLE:
                logger.warning("Milvus package not available. Install with: pip install langchain-milvus pymilvus")
                return False
            
            logger.info("Attempting to initialize Milvus vector store...")
            
            # Try to connect to Milvus
            connection_args = {
                "host": settings.milvus_host,
                "port": settings.milvus_port
            }
        except Exception as e:
            logger.error(f"Failed connecting to MultiVectorStore: {str(e)}")
            raise CustomException(e, sys)




# ============================================================================
# FILE: rag_voice_agent/monitoring/metrics.py
# ============================================================================
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

query_counter = Counter('rag_queries_total', 'Total number of queries')
query_duration = Histogram('rag_query_duration_seconds', 'Query duration')
cache_hits = Counter('rag_cache_hits_total', 'Total cache hits')
cache_misses = Counter('rag_cache_misses_total', 'Total cache misses')
documents_retrieved = Histogram('rag_documents_retrieved', 'Number of documents retrieved')

def track_query(func):
    """Decorator to track query metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        query_counter.inc()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        duration = time.time() - start_time
        query_duration.observe(duration)
        
        if isinstance(result, dict):
            if result.get('from_cache'):
                cache_hits.inc()
            else:
                cache_misses.inc()
            
            if 'sources' in result:
                documents_retrieved.observe(len(result['sources']))
        
        return result
    
    return wrapper



#============================================================================
# FILE: src/utils/exception.py
# ============================================================================
"""
Custom exception handling for the RAG system.
Provides detailed error information with context.
"""
import sys
from typing import Optional

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extract detailed error information including file, line number, and message.
    
    Args:
        error: The exception object
        error_detail: sys module to extract traceback
        
    Returns:
        Formatted error message string
    """
    _, _, exc_tb = error_detail.exc_info()
    
    if exc_tb is None:
        return f"Error: {str(error)}"
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    error_message = f"Error in script: [{file_name}] at line [{line_number}]: {str(error)}"
    return error_message





import sys
import traceback


class CustomException(Exception):
    def __init__(self, error: Exception, error_detail: sys = None):
        self.error = error
        self.error_detail = error_detail
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.error_detail:
            _, _, tb = self.error_detail.exc_info()
            file_name = tb.tb_frame.f_code.co_filename
            line_number = tb.tb_lineno
            return (
                f"Error in script: [{file_name}] "
                f"at line [{line_number}]: {str(self.error)}"
            )
        return str(self.error)


# ============================================================================
# FILE: src/utils/logger.py
# ============================================================================
"""
Centralized logging configuration for the RAG system.
Creates timestamped log files in logs/ directory.
"""
import os
import logging
from datetime import datetime
from pathlib import Path

# Create logs directory
LOG_DIR = os.path.join(os.getcwd(), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Log file with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("[ %(asctime)s ] %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(console_handler)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)



# ============================================================================
# FILE: main.py (SIMPLIFIED TESTING)
# ============================================================================
"""
Main testing script for Multi-Provider RAG System.
Tests document ingestion from S3 and question answering.

Features:
- Load documents from S3 (ll-ai-solutions/data/)
- Multi-provider vector stores (S3 → Milvus → FAISS)
- Multi-provider LLMs (Bedrock Titan → OpenAI → Groq)
- Fast embeddings (Titan Embed v2)
- Comprehensive logging
"""
import sys
from pathlib import Path

import time
from typing import Dict


from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.config.settings import settings
from src.core.ingestion import DocumentIngestion
from src.core.embeddings import MultiProviderEmbeddings
from src.core.vectorstore import MultiVectorStore
from src.core.llm_provider import MultiProviderLLM
from src.config.prompts import PromptTemplates

logger = get_logger(__name__)


def format_time(seconds: float) -> str:
    """Format seconds nicely."""
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    return f"{seconds:.2f} sec"



def print_separator(title: str = ""):
    """Print visual separator."""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)


def test_system_info():
    """Display system configuration."""
    print_separator("SYSTEM CONFIGURATION")
    
    try:
        # Initialize components
        embeddings = MultiProviderEmbeddings()
        vector_store = MultiVectorStore(embeddings)
        llm = MultiProviderLLM()
        
        print(f"\n📊 Configuration:")
        print(f"   Vector Store: {vector_store.store_type}")
        print(f"   Embedding Provider: {embeddings.provider}")
        print(f"   Embedding Model: {settings.embedding_model}")
        print(f"   LLM Provider: {llm.provider}")
        print(f"   LLM Model: {llm.current_model}")
        print(f"   S3 Bucket: {settings.aws_s3_bucket}")
        print(f"   S3 Data Path: {settings.aws_s3_data_prefix}")
        
        print("\n✅ All systems operational!")
        
    except Exception as e:
        logger.error(f"System info failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")


def test_ingest_from_s3():
    """Test document ingestion from S3."""
    print_separator("S3 DOCUMENT INGESTION")
    
    try:
        print(f"\n📁 Loading documents from S3...")
        print(f"   Bucket: {settings.aws_s3_bucket}")
        print(f"   Prefix: {settings.aws_s3_data_prefix}")
        
        # Initialize ingestion with S3
        ingestion = DocumentIngestion(load_from_s3=True)
        
        # Process documents from S3
        chunks = ingestion.process_documents(from_s3=True)
        
        print(f"\n✅ Successfully processed {len(chunks)} document chunks from S3")
        
        # Initialize vector store and embeddings
        print(f"\n💾 Creating embeddings and storing in vector database...")
        embeddings = MultiProviderEmbeddings()
        vector_store = MultiVectorStore(embeddings)
        
        # Add to vector store
        ids = vector_store.add_documents(chunks)
        
        print(f"✅ Added {len(ids)} chunks to {vector_store.store_type} vector store")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"S3 ingestion failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        raise CustomException(e, sys)


def test_ingest_local(directory: str):
    """Test document ingestion from local directory."""
    print_separator(f"LOCAL DOCUMENT INGESTION: {directory}")
    
    try:
        # Initialize ingestion
        ingestion = DocumentIngestion()
        
        # Process documents
        chunks = ingestion.process_documents(source_path=directory)
        
        print(f"\n✅ Successfully processed {len(chunks)} document chunks")
        
        # Initialize vector store and embeddings
        print(f"\n💾 Creating embeddings and storing in vector database...")
        embeddings = MultiProviderEmbeddings()
        vector_store = MultiVectorStore(embeddings)
        
        # Add to vector store
        ids = vector_store.add_documents(chunks)
        
        print(f"✅ Added {len(ids)} chunks to {vector_store.store_type} vector store")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Local ingestion failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        raise CustomException(e, sys)


def test_query(vector_store: MultiVectorStore, question: str):
    """Test question answering with timing."""
    print_separator(f"QUERY: {question}")
    
    try:
        overall_start = time.perf_counter()

        # ---------------------------------------------------
        # 1️⃣ Vector Search Timing
        # ---------------------------------------------------
        print(f"\n🔍 Searching for relevant documents...")
        search_start = time.perf_counter()

        docs = vector_store.similarity_search(
            question, 
            k=settings.top_k
        )

        search_time = time.perf_counter() - search_start
        print(f"   Found {len(docs)} relevant documents")
        print(f"   ⏱️ Retrieval Time: {format_time(search_time)}")

        if not docs:
            print("❌ No relevant documents found")
            return

        # ---------------------------------------------------
        # 2️⃣ Context Preparation
        # ---------------------------------------------------
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])

        # ---------------------------------------------------
        # 3️⃣ LLM Generation Timing
        # ---------------------------------------------------
        print(f"\n🤖 Generating answer...")
        llm = MultiProviderLLM()

        if llm.provider == "bedrock" and "titan" in llm.current_model.lower():
            prompt = PromptTemplates.TITAN_OPTIMIZED_PROMPT.format(
                context=context,
                question=question
            )
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": PromptTemplates.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": PromptTemplates.RAG_PROMPT.format(
                        context=context,
                        question=question
                    )
                }
            ]

        llm_start = time.perf_counter()
        answer = llm.generate(messages)
        llm_time = time.perf_counter() - llm_start

        # ---------------------------------------------------
        # 4️⃣ Total Time
        # ---------------------------------------------------
        total_time = time.perf_counter() - overall_start

        # ---------------------------------------------------
        # 📚 Output
        # ---------------------------------------------------
        print(f"\n📚 Sources:")
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get("filename", "Unknown")
            print(f"   {i}. {filename}")

        print(f"\n💬 Answer:")
        print(f"   {answer}")

        print(f"\n⏱️ Performance Metrics:")
        print(f"   Retrieval Time : {format_time(search_time)}")
        print(f"   LLM Time       : {format_time(llm_time)}")
        print(f"   Total Time     : {format_time(total_time)}")

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        raise CustomException(e, sys)



def interactive_mode():
    """Interactive Q&A mode."""
    print_separator("INTERACTIVE MODE")
    
    try:
        # Initialize vector store
        print("\n⚙️  Initializing system...")
        embeddings = MultiProviderEmbeddings()
        vector_store = MultiVectorStore(embeddings)
        
        print("\n✅ System ready!")
        print("\nCommands:")
        print("  - Type your question")
        print("  - 'exit' or 'quit' to exit")
        
        while True:
            print("\n" + "-"*80)
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            
            if question:
                test_query(vector_store, question)
                
    except Exception as e:
        logger.error(f"Interactive mode failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")


def main():
    """Main function."""
    print_separator("MULTI-PROVIDER RAG SYSTEM")
    
    try:
        while True:
            print_separator("MAIN MENU")
            print("\nOptions:")
            print("1. Show system configuration")
            print("2. Ingest documents from S3 (ll-ai-solutions/data/)")
            print("3. Ingest documents from local directory")
            print("4. Test single query")
            print("5. Interactive Q&A mode")
            print("6. Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                test_system_info()
            
            elif choice == '2':
                test_ingest_from_s3()
            
            elif choice == '3':
                directory = input("Enter directory path: ").strip()
                if Path(directory).exists():
                    test_ingest_local(directory)
                else:
                    print(f"❌ Directory not found: {directory}")
            
            elif choice == '4':
                try:
                    embeddings = MultiProviderEmbeddings()
                    vector_store = MultiVectorStore(embeddings)
                    question = input("Enter your question: ").strip()
                    if question:
                        test_query(vector_store, question)
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
            
            elif choice == '5':
                interactive_mode()
            
            elif choice == '6':
                print_separator()
                print("👋 Thank you for using RAG System!")
                print("="*80 + "\n")
                break
            
            else:
                print("❌ Invalid option")
                
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"Main failed: {str(e)}")
        print(f"\n❌ Fatal error: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
        
        
        