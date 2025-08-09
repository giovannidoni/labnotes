"""
Embedding functionality for content similarity detection.
Uses lightweight sentence-transformers models or OpenAI embeddings for efficient embedding generation.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import os

from sentence_transformers import SentenceTransformer
import openai


logger = logging.getLogger(__name__)

# Global embedding service instances
_embedding_service = None
_openai_embedding_service = None


def is_openai_model(model_name: str) -> bool:
    """Determine if model is an OpenAI model based on name."""
    openai_models = {
        'text-embedding-3-small',
        'text-embedding-3-large', 
        'text-embedding-ada-002'
    }
    return model_name in openai_models


class OpenAIEmbeddingService:
    """Service for generating embeddings using OpenAI's text-embedding models."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """Initialize the OpenAI embedding service."""
        self.model_name = model_name
        self.client = None
        self.initialized = False
    
    def _ensure_initialized(self) -> bool:
        """Lazy initialize the OpenAI client."""
        if self.initialized:
            return True

        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("OPENAI_API_KEY environment variable not set")
                return False
            
            logger.info(f"Initializing OpenAI embedding service with model: {self.model_name}")
            self.client = openai.OpenAI(api_key=api_key)
            self.initialized = True
            logger.info(f"Successfully initialized OpenAI embedding service")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embedding service: {e}")
            return False
    
    def _extract_text_for_embedding(self, item: Dict[str, Any]) -> str:
        """Extract and prepare text from item for embedding generation."""
        # Combine title, content, and full_content for comprehensive embedding
        title = item.get("title", "") or ""
        content = item.get("content", "") or ""
        full_content = item.get("full_content", "") or ""
        
        # Use the longer content field and combine with title
        main_content = full_content if len(full_content) > len(content) else content
        
        # Combine title and content, but avoid excessive duplication
        if title and not title.lower() in main_content.lower()[:200]:
            text = f"{title}. {main_content}"
        else:
            text = main_content or title
        
        # OpenAI has higher token limits, but still truncate for efficiency
        text = text[:8000]  # Roughly 6000-8000 tokens
        
        return text.strip()
    
    async def generate_embedding(self, item: Dict[str, Any]) -> Optional[np.ndarray]:
        """Generate embedding for a single item asynchronously."""
        if not self._ensure_initialized():
            return None
        
        try:
            text = self._extract_text_for_embedding(item)
            if not text:
                logger.debug(f"No text found for embedding: {item.get('title', 'Unknown')}")
                return None
            
            # Run OpenAI API call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.embeddings.create(
                    input=text,
                    model=self.model_name
                )
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            logger.debug(f"Generated OpenAI embedding for '{item.get('title', 'Unknown')[:50]}...': shape={embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to generate OpenAI embedding for '{item.get('title', 'Unknown')[:50]}...': {e}")
            return None
    
    async def generate_embeddings_batch(self, items: List[Dict[str, Any]]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for multiple items concurrently."""
        if not self._ensure_initialized():
            return [None] * len(items)
        
        logger.info(f"Generating OpenAI embeddings for {len(items)} items...")
        
        # Generate embeddings concurrently
        tasks = [self.generate_embedding(item) for item in items]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        result_embeddings = []
        successful = 0
        for i, embedding in enumerate(embeddings):
            if isinstance(embedding, Exception):
                logger.warning(f"OpenAI embedding generation failed for item {i}: {embedding}")
                result_embeddings.append(None)
            elif embedding is not None:
                result_embeddings.append(embedding)
                successful += 1
            else:
                result_embeddings.append(None)
        
        logger.info(f"Successfully generated {successful}/{len(items)} OpenAI embeddings")
        return result_embeddings


class EmbeddingService:
    """Service for generating embeddings from text content using lightweight models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding service."""
        self.model_name = model_name
        self.model = None
        self.initialized = False
    
    def _ensure_initialized(self) -> bool:
        """Lazy initialize the model to avoid loading it unless needed."""
        if self.initialized:
            return True
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.initialized = True
            logger.info(f"Successfully loaded embedding model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            return False
    
    def _extract_text_for_embedding(self, item: Dict[str, Any]) -> str:
        """Extract and prepare text from item for embedding generation."""
        # Combine title, content, and full_content for comprehensive embedding
        title = item.get("title", "") or ""
        content = item.get("content", "") or ""
        full_content = item.get("full_content", "") or ""
        
        # Use the longer content field and combine with title
        main_content = full_content if len(full_content) > len(content) else content
        
        # Combine title and content, but avoid excessive duplication
        if title and not title.lower() in main_content.lower()[:200]:
            text = f"{title}. {main_content}"
        else:
            text = main_content or title
        
        # Truncate to reasonable length for embedding (models typically have token limits)
        # Most sentence-transformers models work well with 200-500 tokens
        text = text[:2000]  # Roughly 300-500 tokens depending on the text
        
        return text.strip()
    
    async def generate_embedding(self, item: Dict[str, Any]) -> Optional[np.ndarray]:
        """Generate embedding for a single item asynchronously."""
        if not self._ensure_initialized():
            return None
        
        try:
            text = self._extract_text_for_embedding(item)
            if not text:
                logger.debug(f"No text found for embedding: {item.get('title', 'Unknown')}")
                return None
            
            # Run embedding generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                self.model.encode,
                text,
                {"convert_to_numpy": True, "normalize_embeddings": True}
            )
            
            logger.debug(f"Generated embedding for '{item.get('title', 'Unknown')[:50]}...': shape={embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to generate embedding for '{item.get('title', 'Unknown')[:50]}...': {e}")
            return None
    
    async def generate_embeddings_batch(self, items: List[Dict[str, Any]]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for multiple items concurrently."""
        if not self._ensure_initialized():
            return [None] * len(items)
        
        logger.info(f"Generating embeddings for {len(items)} items...")
        
        # Generate embeddings concurrently
        tasks = [self.generate_embedding(item) for item in items]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        result_embeddings = []
        successful = 0
        for i, embedding in enumerate(embeddings):
            if isinstance(embedding, Exception):
                logger.warning(f"Embedding generation failed for item {i}: {embedding}")
                result_embeddings.append(None)
            elif embedding is not None:
                result_embeddings.append(embedding)
                successful += 1
            else:
                result_embeddings.append(None)
        
        logger.info(f"Successfully generated {successful}/{len(items)} embeddings")
        return result_embeddings


def get_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> Union[EmbeddingService, OpenAIEmbeddingService, None]:
    """Get or create the appropriate embedding service instance based on model name."""
    global _embedding_service, _openai_embedding_service
    
    try:
        # Auto-detect provider based on model name
        use_openai = is_openai_model(model_name)
        
        if use_openai:
            # Check if we have OpenAI API key
            if not os.getenv('OPENAI_API_KEY'):
                logger.warning(f"OpenAI model '{model_name}' requested but OPENAI_API_KEY not set, falling back to sentence-transformers")
                # Use a default sentence-transformers model as fallback
                model_name = "all-MiniLM-L6-v2"
                use_openai = False
            else:
                if _openai_embedding_service is None or _openai_embedding_service.model_name != model_name:
                    _openai_embedding_service = OpenAIEmbeddingService(model_name)
                return _openai_embedding_service
        
        # Use sentence-transformers (default or fallback)
        if _embedding_service is None or _embedding_service.model_name != model_name:
            _embedding_service = EmbeddingService(model_name)
        return _embedding_service
        
    except Exception as e:
        logger.error(f"Failed to create embedding service: {e}")
        return None
