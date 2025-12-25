# Rag_Forall


# ============================================================================
# FILE: README.md
# ============================================================================
# Multi-Provider RAG System with S3 Integration

Optimized RAG system with multi-provider support for maximum flexibility and cost efficiency.

## 🎯 Key Features

### Vector Stores (Priority Order)
1. **AWS S3 Vectors** ⭐ - Serverless, uses AWS credits
2. **Milvus** - High performance for production
3. **FAISS** - Local fallback, always available

### LLM Providers (Priority Order)
1. **AWS Bedrock** ⭐ - Titan models (fastest, cheapest)
2. **OpenAI** - GPT models
3. **Groq** - Ultra-fast inference

### Embeddings
- **Amazon Titan Embed v2** (1024 dims) ⭐ - Fast & cheap with AWS credits
- **OpenAI Embeddings** - Fallback
- **HuggingFace** - Free fallback

### Document Sources
- **AWS S3**: ll-ai-solutions/data/
- **Local directories**

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Required AWS Setup
```bash
# Set AWS credentials
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1

# S3 buckets
AWS_S3_BUCKET=ll-ai-solutions  # Your document bucket
AWS_S3_DATA_PREFIX=data/  # Folder with documents
AWS_S3_VECTOR_BUCKET=ll-vector-db  # Vector store bucket
```

### 4. Run System
```bash
python main.py
```

## 📋 Menu Options

1. **Show Configuration** - View current providers
2. **Ingest from S3** - Load from ll-ai-solutions/data/
3. **Ingest Local** - Load from local directory
4. **Test Query** - Ask a single question
5. **Interactive Mode** - Continuous Q&A
6. **Exit**

## 🔧 Provider Fallback Logic

### Vector Stores
```
AWS S3 (configured?) → Yes → Use S3 ✅
  ↓ No
Milvus (running?) → Yes → Use Milvus ✅
  ↓ No
FAISS (always works) → Use FAISS ✅
```

### LLMs
```
Bedrock (configured?) → Yes → Try Titan models ✅
  ↓ Titan fails
  → Try Claude models ✅
  ↓ All fail
OpenAI (configured?) → Yes → Use GPT ✅
  ↓ No
Groq (configured?) → Yes → Use Groq ✅
```

## 📊 Optimizations

- **Low Latency**: Titan Express (fastest Bedrock model)
- **Cost Efficient**: Uses AWS credits, cheapest models first
- **High Quality**: Falls back to Claude/GPT when needed
- **Reliability**: Multiple fallbacks ensure system always works