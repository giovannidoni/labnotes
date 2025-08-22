#!/usr/bin/env python3
"""
summarise digest items using AI prompts.
Adds summary and novelty score fields to JSON items.
"""

import argparse
import json
import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import openai
import traceback

from labnotes.utils import setup_logging, save_to_supabase
from labnotes.utils import save_output, load_input, find_most_recent_file
from labnotes.settings import settings

logger = logging.getLogger(__name__)


def map_novelty_score_to_int(score_text: str) -> int:
    """Map novelty score text to integer (1-5)."""
    score_mapping = {"average": 1, "high": 2, "very high": 3}

    # Normalize the input
    normalized_score = score_text.lower().strip()

    # Return mapped value or default to 3 (average)
    return score_mapping.get(normalized_score, 3)


def load_prompt_template(prompt_type) -> str:
    """Load the summarisation prompt template."""
    prompt_file = Path(__file__).parent / "prompts" / f"{prompt_type}.txt"
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Failed to load prompt template from {prompt_file}: {error}")
        return None


class SummarisationService:
    """Service for generating summaries using OpenAI's chat models."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize the summarisation service."""
        self.model_name = model_name
        self.client = None
        self.initialized = False
        self.prompt_template = None

    def _ensure_initialized(self) -> bool:
        """Lazy initialize the OpenAI client."""
        if self.initialized:
            return True

        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY environment variable not set")
                return False

            self.client = openai.OpenAI(api_key=api_key)
            self.prompt_template = load_prompt_template("summarisation")

            if not self.prompt_template:
                logger.error("Failed to load prompt template")
                return False

            self.initialized = True
            logger.info(f"Successfully initialized summarisation service with model: {self.model_name}")
            return True
        except Exception as e:
            error = traceback.format_exc()
            logger.error(f"Failed to initialize summarisation service: {error}")
            return False

    def _extract_content_for_summary(self, item: Dict[str, Any]) -> str:
        """Extract text content from item for summarisation."""
        title = item.get("title", "") or ""
        content = item.get("content", "") or ""
        full_content = item.get("full_content", "") or ""

        # Use the longer content field and combine with title
        main_content = full_content if len(full_content) > len(content) else content

        # Combine title and content
        if title and not title.lower() in main_content.lower()[:200]:
            text = f"Title: {title}\n\nContent: {main_content}"
        else:
            text = f"Content: {main_content or title}"

        return text.strip()

    async def summarise_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate summary and novelty score for a single item."""
        if not self._ensure_initialized():
            return None

        try:
            content = self._extract_content_for_summary(item)
            if not content:
                logger.debug(f"No content found for summarisation: {item.get('title', 'Unknown')}")
                return None

            # Create the prompt
            full_prompt = f"{self.prompt_template}\n\n{content}"

            # Define the JSON schema for structured output
            response_schema = {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of the article in no more than 160 characters",
                    },
                    "_novelty_score": {
                        "type": "string",
                        "enum": ["questionable", "high", "very high"],
                        "description": "Novelty/originality score of the article",
                    },
                },
                "required": ["summary", "_novelty_score"],
                "additionalProperties": False,
            }

            # Run OpenAI API call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.3,
                    max_tokens=200,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "article_summary", "schema": response_schema, "strict": True},
                    },
                ),
            )

            summary_response = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                summary_data = json.loads(summary_response)
                logger.debug(
                    f"Generated summary for '{item.get('title', 'Unknown')[:50]}...': {summary_data.get('summary', '')[:50]}..."
                )
                return summary_data
            except json.JSONDecodeError as e:
                error = traceback.format_exc()
                logger.error(f"Failed to parse summary JSON for '{item.get('title', 'Unknown')[:50]}...': {error}")
                return None

        except Exception as e:
            error = traceback.format_exc()
            logger.error(f"Failed to generate summary for '{item.get('title', 'Unknown')[:50]}...': {error}")
            return None

    async def process_items_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate summaries for multiple items and add fields to JSON."""
        if not self._ensure_initialized():
            return items

        logger.info(f"Generating summaries for {len(items)} items...")

        # Generate summaries concurrently
        tasks = [self.summarise_item(item) for item in items]
        summaries = await asyncio.gather(*tasks, return_exceptions=True)

        # Add summary fields to items
        enhanced_items = []
        successful = 0

        for item, summary in zip(items, summaries):
            enhanced_item = item.copy()

            if isinstance(summary, Exception):
                logger.warning(f"Summarisation failed for item: {summary}")
            elif summary is not None:
                enhanced_item["summary"] = summary.get("summary", "")
                # Map novelty score to integer
                novelty_text = summary.get("_novelty_score", "")
                enhanced_item["_novelty_score"] = map_novelty_score_to_int(novelty_text)
                successful += 1

            enhanced_items.append(enhanced_item)

        logger.info(f"Successfully generated {successful}/{len(items)} summaries")
        return enhanced_items


async def summarise_digest(
    items: List[Dict[str, Any]],
    model_name: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """
    Summarise digest items and add summary fields.
    """
    logger.info(f"Starting summarisation with model={model_name}")

    # Create summarisation service
    summarisation_service = SummarisationService(model_name)

    # Generate summaries and add to items
    enhanced_items = await summarisation_service.process_items_batch(items)

    return enhanced_items


async def main_async():
    """Main async function."""
    parser = argparse.ArgumentParser(description="summarise digest items using AI prompts")
    parser.add_argument("--input", required=True, help="Input JSON digest path")
    parser.add_argument(
        "--section", required=True, help="process only one section/group from feeds (e.g., 'ai_research_and_models')"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name for summarisation (e.g., gpt-4o-mini, gpt-4o, gpt-3.5-turbo)",
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
        input_file = find_most_recent_file(input_path, pattern=f"{args.section}*deduped_*.json")
    except FileNotFoundError as e:
        logger.info("No deduped files found, please run the digest and deduped command first.")
        return 0

    output_file = str(input_file).replace("deduped", "summarised")

    logger.info(f"Input: {input_file}, Output: {output_file}")

    try:
        # Load digest
        items = load_input(str(input_file))

        if not items:
            logger.warning("No items to process")
            return 0

        # summarise
        summarised = await summarise_digest(
            items,
            model_name=args.model,
        )

        # Save results
        save_output(summarised, output_file)

        # Save results
        if len(summarised) > 0 and settings.summarise.save_to_supabase:
            save_to_supabase(summarised, "raw_articles")

        # Report statistics
        summary_count = sum(1 for item in summarised if item.get("summary"))
        logger.info(f"summarisation complete: {summary_count}/{len(summarised)} items summarised")

        return 0

    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"summarisation failed: {error}")
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
