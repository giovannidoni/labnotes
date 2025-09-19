#!/usr/bin/env python3
"""
Deduplicate digest items based on similarity.
Reads digest output, filters similar items keeping the most recent or highest scoring ones.
"""

import argparse
import json
import asyncio
import logging
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from labnotes.tools.utils import setup_logging, find_most_recent_file
from labnotes.similarity import remove_similar_items, parse_date_for_comparison
from labnotes.tools.io import save_output, load_input

logger = logging.getLogger(__name__)


def is_openai_model(model_name: str) -> bool:
    """Determine if model is an OpenAI model based on name."""
    openai_models = {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}
    return model_name in openai_models


def filter_by_score_and_date(
    similar_items: List[Dict[str, Any]], prefer_recent: bool = True, score_weight: float = 0.5
) -> Dict[str, Any]:
    """
    From a list of similar items, select the best one based on score and date.

    Args:
        similar_items: List of similar items to choose from
        prefer_recent: If True, prioritize recent items; if False, prioritize high scores
        score_weight: Weight for score vs recency (0=only date, 1=only score)

    Returns:
        The selected best item
    """
    if not similar_items:
        return None

    if len(similar_items) == 1:
        return similar_items[0]

    # Normalize scores to 0-1 range
    scores = [item.get("_score", 0) for item in similar_items]
    max_score = max(scores) if scores else 1
    min_score = min(scores) if scores else 0
    score_range = max_score - min_score if max_score != min_score else 1

    # Parse dates and normalize to 0-1 range
    dates = [parse_date_for_comparison(item.get("date", "")) for item in similar_items]
    max_date = max(dates) if dates else datetime.now()
    min_date = min(dates) if dates else datetime(1970, 1, 1)
    date_range = (max_date - min_date).total_seconds()
    if date_range == 0:
        date_range = 1

    # Calculate combined score for each item
    best_item = None
    best_combined_score = -1

    for item, date in zip(similar_items, dates):
        # Normalize score (0-1)
        norm_score = (item.get("_score", 0) - min_score) / score_range if score_range else 0

        # Normalize date (0-1, where 1 is most recent)
        norm_date = (date - min_date).total_seconds() / date_range if date_range else 0

        # Calculate combined score
        if prefer_recent:
            # Weight towards recency
            combined_score = norm_date * (1 - score_weight) + norm_score * score_weight
        else:
            # Weight towards score
            combined_score = norm_score * (1 - score_weight) + norm_date * score_weight

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_item = item

        logger.debug(
            f"Item '{item.get('title', 'Unknown')[:50]}...': "
            f"score={item.get('_score', 0)} (norm={norm_score:.2f}), "
            f"date={item.get('date', 'N/A')} (norm={norm_date:.2f}), "
            f"combined={combined_score:.2f}"
        )

    return best_item


async def deduplicate_digest(
    items: List[Dict[str, Any]],
    similarity_threshold: float = 0.85,
    prefer_recent: bool = True,
    score_weight: float = 0.3,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Deduplicate digest items based on similarity.

    Args:
        items: List of digest items
        similarity_threshold: Threshold for considering items similar (0-1)
        prefer_recent: If True, keep most recent; if False, keep highest scoring
        score_weight: Weight for score vs recency when selecting (0=only date, 1=only score)
        model_name: Embedding model to use (OpenAI models auto-detected by name)

    Returns:
        Deduplicated list of items
    """
    # Determine provider based on model name
    use_openai = is_openai_model(model_name)
    provider = "OpenAI" if use_openai else "sentence-transformers"

    logger.info(
        f"Starting deduplication with threshold={similarity_threshold}, "
        f"prefer_recent={prefer_recent}, score_weight={score_weight}, "
        f"model={model_name} (provider: {provider})"
    )

    # Use the similarity module to identify and remove similar items
    # This function already handles grouping and filtering
    deduplicated = await remove_similar_items(
        items,
        similarity_threshold=similarity_threshold,
        model_name=model_name,
    )

    return deduplicated


async def main_async():
    """Main async function."""
    parser = argparse.ArgumentParser(description="Deduplicate digest items based on similarity")
    parser.add_argument("--input", required=True, help="Input JSON digest path")
    parser.add_argument(
        "--section", required=True, help="process only one section/group from feeds (e.g., 'ai_research_and_models')"
    )

    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Similarity threshold (0-1, default: 0.5)")
    parser.add_argument(
        "--prefer-score", action="store_true", help="Prefer high score over recency (default: prefer recent)"
    )
    parser.add_argument(
        "--score-weight",
        type=float,
        default=0.3,
        help="Weight for score vs recency (0=only date, 1=only score, default: 0.3)",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="Embedding model name (OpenAI models: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Set logging level"
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Determine input and output paths
    input_path = Path(args.input) / args.section

    # Find the most recent JSON file in the input path
    try:
        input_file = find_most_recent_file(input_path, pattern=f"{args.section}*digest_*.json")
    except FileNotFoundError as e:
        logger.info("No digest files found, please run the digest command first.")
        return 0

    output_file = str(input_file).replace("digest", "deduped")

    logger.info(f"Input: {input_file}, Output: {output_file}")

    try:
        # Load digest
        items = load_input(str(input_file))

        if not items:
            logger.warning("No items to process")
            return 0

        # Deduplicate
        deduplicated = await deduplicate_digest(
            items,
            similarity_threshold=args.threshold,
            prefer_recent=not args.prefer_score,
            score_weight=args.score_weight,
            model_name=args.model,
        )

        # Save results
        save_output(deduplicated, output_file)

        # Report statistics
        removed_count = len(items) - len(deduplicated)
        logger.info(
            f"Deduplication complete: {len(deduplicated)} items retained, {removed_count} similar items removed"
        )

        return 0

    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Deduplication failed: {error}")
        return 1


def main():
    """Synchronous CLI entry point wrapper."""
    try:
        exit_code = asyncio.run(main_async())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Unexpected error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
