from typing import Dict, Any, Optional, List, Iterator
from src.config.settings import settings
from src.config.prompts import PromptTemplates
from src.core.retriever import Retriever
from src.core.reranker import Reranker
from src.core.cache import Cache
from src.core.llm_provider import get_llm_provider

class RAGPipeline:
    """RAG Pipeline - Main orchestration class."""
    
    def __init__(
        self,
        retriever: Retriever = None,
        reranker: Reranker = None,
        cache: Cache = None
    ):
        self.retriever = retriever or Retriever()
        self.reranker = reranker or Reranker()
        self.cache = cache or Cache()
        self.llm_provider = get_llm_provider()
    
    def query(
        self,
        question: str,
        k: int = None,
        use_cache: bool = True,
        filters: Optional[Dict] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Execute RAG pipeline query."""
        # Check cache
        if use_cache:
            cached = self.cache.get(question, k=k, filters=filters)
            if cached:
                cached['from_cache'] = True
                return cached
        
        # Retrieve documents
        documents = self.retriever.retrieve(question, k=k, filters=filters)
        
        if not documents:
            result = {
                'answer': "I don't have any relevant information to answer that question.",
                'sources': [],
                'from_cache': False
            }
            if use_cache:
                self.cache.set(question, result, k=k, filters=filters)
            return result
        
        # Rerank documents
        if self.reranker.enabled:
            documents = self.reranker.rerank(question, documents)
        
        # Format context
        context = self.retriever.format_context(documents)
        
        # Generate answer
        system_prompt = PromptTemplates.SYSTEM_PROMPT
        user_prompt = PromptTemplates.RAG_PROMPT.format(context=context, question=question)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if stream:
            return self._stream_response(messages, documents)
        
        answer = self.llm_provider.generate(
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens
        )
        
        result = {
            'answer': answer,
            'sources': [
                {
                    'filename': doc['metadata'].get('filename', 'Unknown'),
                    'score': doc.get('score', 0),
                    'rerank_score': doc.get('rerank_score')
                }
                for doc in documents
            ],
            'from_cache': False
        }
        
        if use_cache:
            self.cache.set(question, result, k=k, filters=filters)
        
        return result
    
    def _stream_response(self, messages: List[Dict], documents: List[Dict]) -> Iterator[Dict]:
        """Stream response generator."""
        stream = self.llm_provider.generate(
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens,
            stream=True
        )
        
        for chunk in stream:
            yield {
                'chunk': chunk,
                'sources': [
                    {
                        'filename': doc['metadata'].get('filename', 'Unknown'),
                        'score': doc.get('score', 0)
                    }
                    for doc in documents
                ]
            }
