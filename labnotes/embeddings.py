"""
Embedding functionality for content similarity detection.
Uses LiteLLM for provider-agnostic embedding generation (supports Gemini, OpenAI, and more).
"""

import asyncio
import logging
import traceback
from typing import Any, Dict, List, Optional, Union

import numpy as np

# from sentence_transformers import SentenceTransformer
from litellm import aembedding

logger = logging.getLogger(__name__)

# Global embedding service instance
_litellm_embedding_service = None


def is_gemini_model(model_name: str) -> bool:
    """Determine if model is a Gemini model based on name prefix."""
    return model_name.startswith("gemini/")


def is_openai_model(model_name: str) -> bool:
    """Determine if model is an OpenAI model based on name."""
    openai_prefixes = ["openai/", "text-embedding-"]
    openai_models = {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}
    return model_name in openai_models or any(model_name.startswith(p) for p in openai_prefixes)


class LiteLLMEmbeddingService:
    """Service for generating embeddings using LiteLLM (supports multiple providers)."""

    def __init__(self, model_name: str):
        """Initialize the embedding service."""
        self.model_name = model_name
        self.initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy initialize the embedding service."""
        if self.initialized:
            return True

        try:
            # LiteLLM auto-detects API keys from environment variables
            # GEMINI_API_KEY for Gemini, OPENAI_API_KEY for OpenAI, etc.
            logger.info(f"Initializing embedding service with model: {self.model_name}")
            self.initialized = True
            logger.info("Successfully initialized embedding service")
            return True
        except Exception:
            error = traceback.format_exc()
            logger.error(f"Failed to initialize embedding service: {error}")
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
        if title and title.lower() not in main_content.lower()[:200]:
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

            # Use native async LiteLLM call
            response = await aembedding(
                model=self.model_name,
                input=[text],  # LiteLLM expects a list
            )

            embedding = np.array(response.data[0]["embedding"], dtype=np.float32)

            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)

            logger.debug(f"Generated embedding for '{item.get('title', 'Unknown')[:50]}...': shape={embedding.shape}")
            return embedding

        except Exception:
            error = traceback.format_exc()
            logger.warning(f"Failed to generate embedding for '{item.get('title', 'Unknown')[:50]}...': {error}")
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


def get_embedding_service(
    model_name: str,
) -> Union[LiteLLMEmbeddingService, None]:
    """Get or create the embedding service instance."""
    global _litellm_embedding_service

    try:
        # LiteLLM handles all providers - just create/reuse the service
        if _litellm_embedding_service is None or _litellm_embedding_service.model_name != model_name:
            _litellm_embedding_service = LiteLLMEmbeddingService(model_name)
        return _litellm_embedding_service

    except Exception:
        error = traceback.format_exc()
        logger.error(f"Failed to create embedding service: {error}")
        return None
