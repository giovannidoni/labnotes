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
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from labnotes.utils import setup_logging
from labnotes.similarity import remove_similar_items, parse_date_for_comparison

logger = logging.getLogger(__name__)


async def main_async():
    """Main async function."""
    parser = argparse.ArgumentParser(description="Deduplicate digest items based on similarity")
    parser.add_argument("--input", required=True, help="Input JSON digest path")
    parser.add_argument("--section", required=True, help="process only one section/group from feeds (e.g., 'ai_research_and_models')")

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
        input_file = find_most_recent_file(input_path)
    except FileNotFoundError as e:
        logger.info("No digest files found, please run the digest command first.")
        return 0

    output_file = str(input_file).replace("digest", "deduped")
    
    logger.info(f"Input: {input_file}, Output: {output_file}")

    try:
        # Load digest
        items = load_digest(str(input_file))

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
        save_deduplicated(deduplicated, output_file)

        # Report statistics
        removed_count = len(items) - len(deduplicated)
        logger.info(
            f"Summarisation complete: {len(deduplicated)} items retained, {removed_count} similar items removed"
        )

        return 0

    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
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
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
