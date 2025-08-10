"""
Similarity-based deduplication functionality.
Removes items that are too similar to each other, keeping the most recent ones.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import numpy as np
import traceback

from labnotes.embeddings import get_embedding_service

logger = logging.getLogger(__name__)


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    try:
        # Embeddings should already be normalized, but ensure it
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Clamp to [0, 1] range and handle numerical precision issues
        similarity = max(0.0, min(1.0, float(similarity)))

        return similarity

    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Failed to calculate cosine similarity: {error}")
        return 0.0


def parse_date_for_comparison(date_str: str) -> datetime:
    """Parse date string for comparison purposes."""
    try:
        if date_str:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    except ValueError:
        try:
            # Try alternative format without time
            return datetime.strptime(date_str[:10], "%Y-%m-%d")
        except ValueError:
            pass

    # Return epoch if parsing fails
    return datetime(1970, 1, 1)


async def remove_similar_items(
    items: List[Dict[str, Any]], similarity_threshold: float = 0.85, model_name: str = "all-MiniLM-L6-v2"
) -> List[Dict[str, Any]]:
    """Remove items that are too similar to each other, keeping the most recent ones."""
    if not items:
        return items

    # Determine provider from model name
    from labnotes.embeddings import is_openai_model

    provider = "OpenAI" if is_openai_model(model_name) else "sentence-transformers"
    logger.info(
        f"Starting similarity-based deduplication on {len(items)} items "
        f"(threshold: {similarity_threshold}, model: {model_name}, provider: {provider})"
    )

    # Get embedding service
    embedding_service = get_embedding_service(model_name)

    # Generate embeddings for all items
    embeddings = await embedding_service.generate_embeddings_batch(items)

    # Filter out items without embeddings and save embeddings to items
    valid_items = []
    valid_embeddings = []
    no_embedding_count = 0

    for item, embedding in zip(items, embeddings):
        if embedding is not None:
            # Create a copy of the item and add the embedding
            item_with_embedding = item.copy()
            item_with_embedding["embedding"] = embedding.tolist()  # Convert numpy array to list for JSON serialization
            valid_items.append(item_with_embedding)
            valid_embeddings.append(embedding)
        else:
            no_embedding_count += 1

    if no_embedding_count > 0:
        logger.warning(f"Could not generate embeddings for {no_embedding_count} items")

    if len(valid_items) <= 1:
        logger.info("Not enough items with embeddings for similarity comparison")
        # Return items with embeddings preserved
        final_items = []
        for item, embedding in zip(items, embeddings):
            if embedding is not None:
                item_with_embedding = item.copy()
                # item_with_embedding['embedding'] = embedding.tolist()
                final_items.append(item_with_embedding)
            else:
                final_items.append(item)
        return final_items

    logger.info(f"Comparing {len(valid_items)} items with valid embeddings")

    # Find similar pairs
    similar_pairs = []

    for i in range(len(valid_items)):
        for j in range(i + 1, len(valid_items)):
            similarity = calculate_cosine_similarity(valid_embeddings[i], valid_embeddings[j])

            if similarity >= similarity_threshold:
                similar_pairs.append((i, j, similarity))
                logger.debug(
                    f"Found similar items: '{valid_items[i].get('title', 'Unknown')[:50]}...' "
                    f"and '{valid_items[j].get('title', 'Unknown')[:50]}...' "
                    f"(similarity: {similarity:.3f})"
                )

    if not similar_pairs:
        logger.info("No similar items found")
        # Return items with embeddings preserved
        final_items = []
        for item, embedding in zip(items, embeddings):
            if embedding is not None:
                item_with_embedding = item.copy()
                # item_with_embedding['embedding'] = embedding.tolist()
                final_items.append(item_with_embedding)
            else:
                final_items.append(item)
        return final_items

    logger.info(f"Found {len(similar_pairs)} similar item pairs")

    # Build sets of similar items (connected components)
    similar_groups = []
    processed = set()

    for i, j, similarity in similar_pairs:
        if i in processed and j in processed:
            continue

        # Find existing group that contains either item
        target_group = None
        for group in similar_groups:
            if i in group or j in group:
                target_group = group
                break

        if target_group is None:
            # Create new group
            similar_groups.append({i, j})
        else:
            # Add to existing group
            target_group.add(i)
            target_group.add(j)

        processed.add(i)
        processed.add(j)

    # For each group, keep only the most recent item
    items_to_keep = set(range(len(valid_items)))  # Start with all items
    removed_count = 0

    for group in similar_groups:
        if len(group) <= 1:
            continue

        # Find the most recent item in the group
        group_items = [(idx, valid_items[idx]) for idx in group]

        # Sort by date (most recent first)
        group_items.sort(key=lambda x: parse_date_for_comparison(x[1].get("date", "")), reverse=True)

        # Keep only the most recent item
        most_recent_idx = group_items[0][0]
        items_to_remove = [idx for idx, _ in group_items[1:]]

        for idx in items_to_remove:
            items_to_keep.discard(idx)
            removed_count += 1

        logger.debug(
            f"In similarity group, keeping most recent: "
            f"'{valid_items[most_recent_idx].get('title', 'Unknown')[:50]}...' "
            f"(removed {len(items_to_remove)} similar items)"
        )

    # Create final list with kept items
    final_items = []

    # Add items that had valid embeddings and were kept
    for idx in sorted(items_to_keep):
        valid_item = {k: v for k, v in valid_items[idx].items() if k != "embedding"}  # Exclude embedding
        final_items.append(valid_item)

    # Add items that couldn't get embeddings (preserve them)
    for item, embedding in zip(items, embeddings):
        if embedding is None:
            final_items.append(item)

    logger.info(
        f"Similarity-based deduplication complete: {len(final_items)} items retained, "
        f"{removed_count} similar items removed"
    )

    return final_items
