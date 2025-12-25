import sentence_transformers
import faiss
import groq
from pathlib import Path

from src.Rag  import LightningRAG
from src.Rag import print_separator, format_time



def main():
    """Auto-start Lightning RAG with preloaded data."""
    print_separator("AUTO SETUP")

    DATA_DIR = "data"

    # Initialize RAG
    rag = LightningRAG()

    # Auto-ingest if index is empty
    if rag.vector_store.index is None or rag.vector_store.index.ntotal == 0:
        print(f"\n📥 No existing vector index found.")
        print(f"📂 Ingesting documents from '{DATA_DIR}' ...")

        if not Path(DATA_DIR).exists():
            raise RuntimeError(f"❌ Data directory not found: {DATA_DIR}")

        rag.ingest_documents(DATA_DIR)

        print("✅ Ingestion complete. Vector DB ready.")
    else:
        print("✅ Existing vector index loaded. Skipping ingestion.")

    # Start interactive Q&A immediately
    print_separator("READY - ASK QUESTIONS")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("❓ Your question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        result = rag.query(question)

        print(f"\n💬 Answer: {result['answer']}")
        print(
            f"⏱️  Time: {format_time(result['total_time_ms'])} "
            f"{'(cached)' if result['from_cache'] else ''}"
        )
        print("-" * 80)






    
   