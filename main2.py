# =============================================================================
# OPTIMIZED MAIN - main.py
# =============================================================================
"""
Ultra-fast RAG query with caching and optimizations.
"""
import time
from typing import Dict
from src2.cache import OptimizedCache

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