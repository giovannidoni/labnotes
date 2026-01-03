#!/usr/bin/env python3
import argparse
import asyncio
import datetime
import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import feedparser
import yaml
from firecrawl import FirecrawlApp
from jinja2 import Environment, FileSystemLoader, select_autoescape

from labnotes.scoring import scoring
from labnotes.scraping import ScrapingMethod, domain_of, scrape_article_content
from labnotes.settings import settings
from labnotes.tools.io import write_file
from labnotes.tools.utils import setup_logging

# Set up logging
logger = logging.getLogger(__name__)


def clean_html(s: str) -> str:
    cleaned = re.sub("<[^<]+?>", "", s or "")
    logger.debug(f"Cleaned HTML: {len(s or '')} -> {len(cleaned)} chars")
    return cleaned


def extract_date(entry) -> str:
    """Extract publication date from feed entry and return as ISO string."""
    dt = None
    for k in ["published_parsed", "updated_parsed"]:
        if k in entry and entry[k]:
            dt = datetime.datetime(*entry[k][:6])
            break
    if dt is None and getattr(entry, "published_parsed", None):
        dt = datetime.datetime(*entry.published_parsed[:6])

    if dt:
        return dt.strftime("%Y-%m-%d %H:%M")
    return ""


def within_hours(entry, hours: int) -> bool:
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
    cutoff = cutoff.replace(tzinfo=None)
    dt = None
    for k in ["published_parsed", "updated_parsed"]:
        if k in entry and entry[k]:
            dt = datetime.datetime(*entry[k][:6])
            break
    if dt is None and getattr(entry, "published_parsed", None):
        dt = datetime.datetime(*entry.published_parsed[:6])

    within_range = (dt is not None) and (dt >= cutoff)
    if dt:
        logger.debug(f"Entry date: {dt}, cutoff: {cutoff}, within range: {within_range}")
    else:
        logger.debug(f"No valid date found for entry, excluding from results")
    return within_range


def extract_content(entry) -> str:
    """Extract content from feed entry, preferring content over summary."""
    content = ""

    # Try to get content from various possible fields
    if hasattr(entry, "content") and entry.content:
        if isinstance(entry.content, list) and len(entry.content) > 0:
            content = entry.content[0].get("value", "")
        else:
            content = str(entry.content)
    elif hasattr(entry, "description") and entry.description:
        content = entry.description
    elif hasattr(entry, "summary") and entry.summary:
        content = entry.summary

    return clean_html(content)


def extract_original_source(link: str, feed_url: str, entry) -> str:
    """Extract the original source domain from a feed entry."""
    # First try to get domain from the link
    source = domain_of(link)

    # If link is from an aggregator, try to find original source in entry metadata
    if "takara.ai" in source or "aggregator" in source.lower():
        # Check for source information in entry attributes
        if hasattr(entry, "source") and hasattr(entry.source, "href"):
            original_source = domain_of(entry.source.href)
            if original_source and original_source != source:
                return original_source

        # Check for author field that might contain source
        if hasattr(entry, "author") and entry.author:
            # Sometimes aggregators put the source in the author field
            author = entry.author.lower()
            # Look for common domain patterns in author field
            import re

            domain_match = re.search(r"([a-zA-Z0-9-]+\.[a-zA-Z]{2,})", author)
            if domain_match:
                return domain_match.group(1)

    return source


async def fetch_feed_items(
    session: aiohttp.ClientSession,
    group: str,
    url: str,
    hours: int,
    scraping_method: ScrapingMethod,
    firecrawl_app: Optional[FirecrawlApp] = None,
) -> List[Dict[str, Any]]:
    """Fetch items from a single feed asynchronously."""
    logger.info(f"Fetching feed: {group} from {url}")
    items = []
    try:
        # Fetch feed content
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            feed_content = await response.text()

        logger.debug(f"Downloaded feed content: {len(feed_content)} chars")
        d = feedparser.parse(feed_content)
        logger.info(f"Parsed feed with {len(d.entries)} entries")

        # Process entries and scrape content concurrently
        scraping_tasks = []
        valid_entries = []
        all_list = set()

        for e in d.entries:
            if not within_hours(e, hours):
                continue
            link = e.get("link") or e.get("id") or ""
            if not link:
                continue
            valid_entries.append(e)
            scraping_tasks.append(scrape_article_content(session, link, scraping_method, firecrawl_app))

        if not valid_entries:
            logger.info(f"No valid entries found in feed {group}")
            return items

        logger.info(f"Scraping {len(valid_entries)} items from {group} using {scraping_method.value}...")
        scraped_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)

        success_count = 0
        for e, scraped_result in zip(valid_entries, scraped_results):
            if isinstance(scraped_result, Exception):
                logger.warning(f"Scraping failed for item: {e.get('title', 'Unknown')}: {scraped_result}")
                scraped_content, final_url, final_source = "", "", ""
            else:
                scraped_content, final_url, final_source = scraped_result
                if scraped_content:
                    success_count += 1

            link = e.get("link") or e.get("id") or ""

            # Extract and clean feed content (remove HTML tags)
            raw_feed_content = extract_content(e)
            feed_content = clean_html(raw_feed_content)

            # Use the final source from scraping if available, otherwise extract from original
            if final_source:
                source = final_source
                # Use the original article URL if we found one
                if final_url and final_url != link:
                    link = final_url
            else:
                source = extract_original_source(link, url, e)

            if link in all_list:
                logger.debug(f"Skipping duplicate link: {link}")
                continue

            all_list.add(link)
            items.append(
                {
                    "title": (e.get("title") or "").strip(),
                    "link": link,
                    "content": feed_content,
                    "full_content": scraped_content,
                    "source": source,
                    "group": group,
                    "date": extract_date(e),
                }
            )

        logger.info(f"Successfully processed {len(items)} items from {group} ({success_count} with scraped content)")

    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Failed to fetch feed {url}: {error}")

    return items


def is_quality_article(item: Dict[str, Any], min_length: int = 300) -> bool:
    """
    Check if item contains quality article content rather than spam or link collections.

    Uses multiple heuristics to determine content quality:
    - Minimum character count
    - Sentence structure and paragraph detection
    - Link density analysis
    - Word variety and repetition patterns
    - Content coherence indicators
    """
    content = item.get("content", "") or ""
    full_content = item.get("full_content", "") or ""

    # Use the longer of the two content fields
    text = full_content if len(full_content) > len(content) else content
    text = text.strip()

    if len(text) < min_length:
        logger.debug(f"Content too short for '{item.get('title', 'Unknown')[:50]}...': {len(text)} < {min_length}")
        return False

    # Count sentences (basic heuristic: periods, exclamation marks, question marks)
    import re

    sentences = re.split(r"[.!?]+", text)
    sentence_count = len([s for s in sentences if len(s.strip()) > 10])

    # Count paragraphs (double newlines or substantial line breaks)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraph_count = len([p for p in paragraphs if len(p.strip()) > 50])

    # Count URLs and links
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    urls = re.findall(url_pattern, text)
    url_count = len(urls)

    # Count words
    words = text.split()
    word_count = len(words)

    if word_count == 0:
        logger.debug(f"No words found in content for '{item.get('title', 'Unknown')[:50]}...'")
        return False

    # Calculate link density (percentage of text that is URLs)
    url_chars = sum(len(url) for url in urls)
    link_density = url_chars / len(text) if len(text) > 0 else 0

    # Check for excessive repetition (spam indicator)
    word_freq = {}
    for word in words:
        word_lower = word.lower().strip('.,!?;:"()[]{}')
        if len(word_lower) > 3:  # Only count meaningful words
            word_freq[word_lower] = word_freq.get(word_lower, 0) + 1

    # Calculate repetition score
    if word_freq:
        max_freq = max(word_freq.values())
        unique_words = len(word_freq)
        repetition_score = max_freq / unique_words if unique_words > 0 else 1
    else:
        repetition_score = 1

    # Quality scoring heuristics
    quality_indicators = {
        "sufficient_sentences": sentence_count >= 10,
        "has_paragraphs": paragraph_count >= 2,
        "low_link_density": link_density < 0.3,  # Less than 30% links
        "reasonable_length": len(text) >= min_length,
        "word_variety": len(word_freq) > 50,  # At least 20 unique meaningful words
        "low_repetition": repetition_score < 0.3,  # No word appears more than 30% of the time
        "sentence_length": word_count / max(sentence_count, 1) > 5,  # Average 5+ words per sentence
    }

    # Calculate quality score
    quality_score = sum(quality_indicators.values())
    total_indicators = len(quality_indicators)

    # Need at least 70% of quality indicators to pass
    is_quality = quality_score >= (total_indicators * 0.7)

    # Special handling for very short content - be more strict
    if len(text) < min_length * 2:  # Less than 2x minimum
        is_quality = is_quality and quality_score >= (total_indicators * 0.8)

    # Log detailed analysis for debugging
    logger.debug(
        f"Quality analysis for '{item.get('title', 'Unknown')[:50]}...': "
        f"length={len(text)}, sentences={sentence_count}, paragraphs={paragraph_count}, "
        f"urls={url_count}, link_density={link_density:.2f}, repetition={repetition_score:.2f}, "
        f"unique_words={len(word_freq)}, quality_score={quality_score}/{total_indicators}, "
        f"is_quality={is_quality}"
    )

    if not is_quality:
        logger.debug(f"Quality indicators failed: {[k for k, v in quality_indicators.items() if not v]}")

    return is_quality


async def fetch_all(
    feeds: Dict[str, list], hours: int, scraping_method: ScrapingMethod, section: Optional[str] = None
) -> List[Dict[str, Any]]:
    logger.info(f"Starting feed processing with {hours}h lookback, method: {scraping_method.value}")

    # Filter feeds to only the specified section if provided
    if section:
        if section not in feeds:
            logger.error(f"Section '{section}' not found in feeds. Available sections: {list(feeds.keys())}")
            return []
        feeds = {section: feeds[section]}
        logger.info(f"Processing only section: {section}")

    # Initialize Firecrawl if needed and API key is available
    firecrawl_app = None
    if scraping_method == ScrapingMethod.FIRECRAWL:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if api_key:
            try:
                firecrawl_app = FirecrawlApp(api_key=api_key)
                logger.info("Using Firecrawl for web scraping")
            except Exception as e:
                error = traceback.format_exc()
                logger.warning(f"Failed to initialize Firecrawl: {error}")
                logger.info("Falling back to newspaper3k scraping")
                scraping_method = ScrapingMethod.NEWSPAPER
        else:
            logger.warning("FIRECRAWL_API_KEY not set, falling back to newspaper3k")
            scraping_method = ScrapingMethod.NEWSPAPER

    logger.info(f"Final scraping method: {scraping_method.value}")

    async with aiohttp.ClientSession() as session:
        # Create tasks for all feeds
        feed_tasks = []
        for group, urls in feeds.items():
            for url in urls:
                feed_tasks.append(fetch_feed_items(session, group, url, hours, scraping_method, firecrawl_app))

        # Fetch all feeds concurrently
        logger.info(f"Fetching {len(feed_tasks)} feeds concurrently...")
        feed_results = await asyncio.gather(*feed_tasks, return_exceptions=True)

        # Flatten results
        items = []
        failed_feeds = 0
        for result in feed_results:
            if isinstance(result, Exception):
                logger.error(f"Feed processing failed: {result}")
                failed_feeds += 1
                continue
            items.extend(result)

    logger.info(f"Feed processing complete: {len(items)} items collected, {failed_feeds} feeds failed")

    # Dedupe by link
    seen = set()
    deduped = []
    duplicates = 0
    for it in items:
        if it["link"] in seen:
            duplicates += 1
            continue
        seen.add(it["link"])
        deduped.append(it)

    logger.info(f"Deduplication complete: {len(deduped)} unique items, {duplicates} duplicates removed")

    # Filter items with insufficient content quality
    quality_filtered = []
    low_quality = 0
    for item in deduped:
        if is_quality_article(item):
            quality_filtered.append(item)
        else:
            low_quality += 1

    logger.info(
        f"Quality filtering complete: {len(quality_filtered)} quality articles retained, "
        f"{low_quality} items filtered out for low content quality"
    )

    return quality_filtered


async def render_outputs(items: List[Dict[str, Any]], section: str, out_dir: str) -> None:
    out_dir = out_dir.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
    base = f"{out_dir}/{section}_digest_{ts}"
    env = Environment(loader=FileSystemLoader(searchpath="labnotes/templates"), autoescape=select_autoescape())

    # Prepare all outputs
    output_tasks = []

    json_content = json.dumps(items, indent=2, ensure_ascii=False)
    output_tasks.append(write_file(base + ".json", json_content))

    # Write all files concurrently
    if output_tasks:
        await asyncio.gather(*output_tasks)
        logger.info(f"Successfully wrote {len(output_tasks)} output files")
    else:
        logger.warning("No output formats specified")


async def main_async():
    parser = argparse.ArgumentParser(description="AI Daily Digest (async)")
    parser.add_argument("--out", default="./out", help="output directory")
    parser.add_argument("--section", help="process only one section/group from feeds (e.g., 'ai_research_and_models')")
    parser.add_argument(
        "--scraper",
        choices=["trafilatura", "newspaper", "beautifulsoup", "firecrawl", "auto"],
        default="trafilatura",
        help="Web scraping method (default: trafilatura)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Setup logging with the specified level using external module
    setup_logging(args.log_level)

    logger.info("Starting AI Daily Digest")
    logger.debug(f"Arguments: {vars(args)}")

    try:
        with open(settings.feed_file, "r", encoding="utf-8") as f:
            feeds = yaml.safe_load(f)
        logger.info(f"Loaded {len(feeds)} feed groups from {settings.feed_file}")

        with open(settings.keywords_file, "r", encoding="utf-8") as f:
            kw = json.load(f)
        logger.info(f"Loaded keywords configuration from {settings.keywords_file}")
    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Failed to load configuration files: {error}")
        return 1

    scraping_method = ScrapingMethod(args.scraper)
    items = await fetch_all(feeds, hours=settings.digest.hours, scraping_method=scraping_method, section=args.section)

    top = scoring(items, kw["audiences"], settings.digest.top)

    out_dir = Path(args.out) / args.section if getattr(args, "section") else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    await render_outputs(top, args.section, str(out_dir))

    logger.info(f"Digest generation complete! Wrote digest with {len(top)} items to {out_dir}")
    return 0


def main():
    """Synchronous CLI entry point wrapper for the async main function."""
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
