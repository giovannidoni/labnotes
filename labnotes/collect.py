#!/usr/bin/env python3
"""
Deduplicate digest items based on similarity.
Reads digest output, filters similar items keeping the most recent or highest scoring ones.
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from litellm import completion

from labnotes.settings import settings
from labnotes.summarise import load_prompt_template
from labnotes.tools.io import load_input, save_output
from labnotes.tools.utils import find_most_recent_file, get_feed_sections, setup_logging

logger = logging.getLogger(__name__)


class ResultSummarisationService:
    """Service for generating result summaries using LiteLLM (supports multiple providers)."""

    def __init__(self, model_name: str):
        """Initialize the summarisation service."""
        self.model_name = model_name
        self.initialized = False
        self.prompt_template = None

    def _ensure_initialized(self) -> bool:
        """Lazy initialize the summarisation service."""
        if self.initialized:
            return True

        try:
            # LiteLLM auto-detects API keys from environment variables
            # GEMINI_API_KEY for Gemini, OPENAI_API_KEY for OpenAI, etc.
            self.prompt_template = load_prompt_template("collect_summaries")

            if not self.prompt_template:
                logger.error("Failed to load prompt template")
                return False

            self.initialized = True
            logger.info(f"Successfully initialized summarisation service with model: {self.model_name}")
            return True
        except Exception:
            error = traceback.format_exc()
            logger.error(f"Failed to initialize summarisation service: {error}")
            return False

    def _prompt_for_item(self, item: Dict[str, Any], index: int) -> str:
        """Extract text content from item for summarisation."""
        title = item.get("title", "") or ""
        summary = item.get("summary", "") or ""
        link = item.get("link", "") or ""

        text = f"Index: {index}\nTitle: {title}\nLink: {link}\nSummary: {summary}"

        return text.strip()

    def _prompt_for_section(self, items: List[Dict[str, Any]], section: str, index: int) -> str:
        """Extract text content from item for summarisation."""
        if not section:
            section = ""
        for item in items:
            content = self._prompt_for_item(item, index)
            index += 1
            if content:
                section += f"{content}\n\n"
        return section.strip()

    def _get_prompt_input(self, section_items: Dict[str, List[Dict[str, Any]]]) -> str:
        """Extract text content from item for summarisation."""
        full_prompt = ""
        index = 0
        for section, items in section_items.items():
            section_header = f"***Section: {section}***\n"
            if section == section:
                full_prompt += self._prompt_for_section(items, section_header, index)
                index += len(items)

        return full_prompt.strip()

    def process_items_batch(self, section_items: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate result summary for a batch of items."""
        if not self._ensure_initialized():
            return None

        try:
            content = self._get_prompt_input(section_items)
            logger.debug(content)

            # Create the prompt
            full_prompt = f"{self.prompt_template}\n\n{content}"

            # Define the JSON schema for structured output
            response_schema = {
                "type": "object",
                "properties": {
                    "picked_headlines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_number": {
                                    "type": "integer",
                                    "description": "The item number of the picked headline",
                                },
                                "summary": {
                                    "type": "string",
                                    "description": "Summary of the article in no more than 160 characters",
                                },
                                "link": {
                                    "type": "string",
                                    "description": "Link to the original article",
                                },
                                "reason_for_choice": {
                                    "type": "string",
                                    "description": "Reason for picking this headline",
                                },
                            },
                            "required": ["item_number", "summary", "link", "reason_for_choice"],
                            "additionalProperties": False,
                        },
                    },
                    "digest": {
                        "type": "string",
                        "description": "Summary of the most important information across other posts",
                    },
                },
                "required": ["picked_headlines", "digest"],
                "additionalProperties": False,
            }

            # Use LiteLLM for provider-agnostic API call
            logger.debug(full_prompt)
            response = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.3,
                max_tokens=2500,
                response_format={
                    "type": "json_object",
                    "response_schema": response_schema,
                },
            )

            summary_response = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                logger.info(f"Generated result summary: {summary_response}")
                summary_data = json.loads(summary_response)
                logger.debug("Generated result summary.")
                return summary_data
            except json.JSONDecodeError:
                error = traceback.format_exc()
                logger.warning(f"Failed to parse result summary JSON: {error}")
                return None

        except Exception:
            error = traceback.format_exc()
            logger.warning(f"Failed to generate result summary: {error}")
            return None


def filter_items_by_score(items, min_score=0.0, min_novelty_score=0.0, filter_mode="OR"):
    """
    Filter items based on manager and engineer score thresholds.
    """
    filtered_items = []
    initial_n = len(items)
    kept = 0

    for item in items:
        score = item.get("_score", 0)
        novelty_score = item.get("_novelty_score", 0)

        # Check filter conditions
        score_filter = score >= min_score
        novelty_filter = novelty_score >= min_novelty_score
        logger.debug(f"Item score: {score}, novelty_score: {novelty_score}")

        if filter_mode == "AND":
            # Both conditions must be met
            if score_filter and novelty_filter:
                item["index"] = kept  # Add index for tracking
                kept += 1
                filtered_items.append(item)
        else:  # OR mode
            # Either condition must be met
            if score_filter or novelty_filter:
                item["index"] = kept  # Add index for tracking
                kept += 1
                filtered_items.append(item)
    logger.info(
        f"Filtered {initial_n} items down to {kept} based on score thresholds (score: {min_score}, novelty: {min_novelty_score})"
    )
    return filtered_items


def summarise_results(
    section_items: List[Dict[str, Any]],
    model_name: str,
) -> List[Dict[str, Any]]:
    """
    Summarise digest items and add summary fields.
    """
    logger.info(f"Starting summarisation with model={model_name}")

    # Create summarisation service
    result_service = ResultSummarisationService(model_name)

    # Generate summaries and add to items
    enhanced_items = result_service.process_items_batch(section_items)

    return enhanced_items


def _main():
    """Main async function."""
    parser = argparse.ArgumentParser(description="summarise digest items using AI prompts")
    parser.add_argument("--input", required=True, help="Input JSON digest path")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Set logging level"
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Determine input and output paths
    sections = get_feed_sections()

    # Find the most recent JSON file in the input path
    input_files = {}
    for section in sections:
        input_files[section] = find_most_recent_file(Path(args.input), pattern=f"{section}/{section}*summarised_*.json")
    if not len(input_files):
        logger.info("No summarised files found.")
        return 0

    section_items = {}

    for section, input_file in input_files.items():
        try:
            # Load digest
            items = load_input(str(input_file))
            items = filter_items_by_score(
                items,
                min_score=settings.collect.min_score,
                min_novelty_score=settings.collect.min_novelty_score,
                filter_mode=settings.collect.filter_mode,
            )

            if not items:
                logger.warning("No items to process")
                continue

            section_items[section] = items

        except Exception:
            error = traceback.format_exc()
            logger.error(f"Failed to process section {section}: {error}")
            continue

    results = summarise_results(
        section_items,
        model_name=settings.collect.model,
    )
    save_output_path = Path(args.input) / "summarised_results.json"
    save_output(results, str(save_output_path))


def main():
    """Synchronous CLI entry point wrapper."""
    try:
        _main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception:
        error = traceback.format_exc()
        logger.error(f"Unexpected error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
