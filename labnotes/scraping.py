"""
Web scraping functionality for content extraction.
Supports multiple scraping methods including newspaper3k, BeautifulSoup, and Firecrawl.
"""

import asyncio
import logging
import re
import traceback
from enum import Enum
from typing import Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import trafilatura
from bs4 import BeautifulSoup
from firecrawl import FirecrawlApp
from newspaper import Article

logger = logging.getLogger(__name__)


class ScrapingMethod(Enum):
    NEWSPAPER = "newspaper"
    BEAUTIFULSOUP = "beautifulsoup"
    FIRECRAWL = "firecrawl"
    TRAFILATURA = "trafilatura"
    AUTO = "auto"


def domain_of(link: str) -> str:
    """Extract domain from URL."""
    try:
        domain = urlparse(link).netloc.replace("www.", "")
        logger.debug(f"Extracted domain: {domain} from {link}")
        return domain
    except Exception as e:
        error = traceback.format_exc()
        logger.warning(f"Failed to extract domain from {link}: {error}")
        return ""


def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file."""
    try:
        parsed = urlparse(url.lower())
        path = parsed.path.lower()
        is_pdf = (
            path.endswith(".pdf")
            or "pdf" in parsed.query.lower()
            or "content-type=application/pdf" in parsed.query.lower()
        )
        logger.debug(f"PDF check for {url}: {is_pdf}")
        return is_pdf
    except Exception as e:
        error = traceback.format_exc()
        logger.warning(f"Error checking if URL is PDF {url}: {error}")
        return False


async def scrape_with_newspaper(session: aiohttp.ClientSession, url: str, timeout: int = 30) -> str:
    """Scrape content using newspaper3k library."""
    logger.debug(f"Starting newspaper scraping for: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            response.raise_for_status()
            html_content = await response.text()

        logger.debug(f"Downloaded HTML content: {len(html_content)} chars")

        # Use newspaper3k to parse the article
        article = Article(url, language="en")
        article.set_html(html_content)
        article.parse()

        # Get the article text
        text = article.text.strip()

        # Add title if available and not already in text
        if article.title and article.title not in text[:200]:
            text = f"{article.title}\n\n{text}"

        result = text[:10000]  # Limit content size
        logger.info(f"Newspaper3k successfully scraped {url}: {len(result)} chars extracted")
        return result

    except Exception as e:
        error = traceback.format_exc()
        logger.warning(f"Newspaper3k failed for {url}: {error}")
        return ""


async def scrape_with_beautifulsoup(session: aiohttp.ClientSession, url: str, timeout: int = 30) -> str:
    """Scrape content using BeautifulSoup (fallback method)."""
    logger.debug(f"Starting BeautifulSoup scraping for: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            response.raise_for_status()
            content = await response.text()

        logger.debug(f"Downloaded HTML content: {len(content)} chars")
        soup = BeautifulSoup(content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()

        # Try to find main content areas (common article selectors)
        content_selectors = [
            "article",
            '[role="main"]',
            ".content",
            ".post-content",
            ".entry-content",
            ".article-content",
            "main",
            ".main-content",
        ]

        content_text = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content_text = content_elem.get_text(separator=" ", strip=True)
                break

        # Fallback to body if no main content found
        if not content_text:
            content_text = soup.body.get_text(separator=" ", strip=True) if soup.body else ""

        # Clean up whitespace and limit size
        content_text = re.sub(r"\s+", " ", content_text).strip()
        result = content_text[:10000]
        logger.info(f"BeautifulSoup successfully scraped {url}: {len(result)} chars extracted")
        return result

    except Exception as e:
        error = traceback.format_exc()
        logger.warning(f"BeautifulSoup failed for {url}: {error}")
        return ""


async def scrape_with_trafilatura(session: aiohttp.ClientSession, url: str, timeout: int = 30) -> str:
    """Scrape content using trafilatura library (modern, well-maintained)."""
    logger.debug(f"Starting trafilatura scraping for: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            response.raise_for_status()
            html_content = await response.text()

        logger.debug(f"Downloaded HTML content: {len(html_content)} chars")

        # Use trafilatura to extract content
        text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True,
        )

        if not text:
            logger.warning(f"Trafilatura extracted no content from {url}")
            return ""

        result = text[:10000]
        logger.info(f"Trafilatura successfully scraped {url}: {len(result)} chars extracted")
        return result

    except Exception as e:
        error = traceback.format_exc()
        logger.warning(f"Trafilatura failed for {url}: {error}")
        return ""


def scrape_with_firecrawl(url: str, firecrawl_app: Optional[FirecrawlApp] = None) -> str:
    """Scrape content using Firecrawl API."""
    if not firecrawl_app:
        logger.debug("No Firecrawl app provided, returning empty string")
        return ""

    logger.debug(f"Starting Firecrawl scraping for: {url}")
    try:
        # Use Firecrawl's scrape endpoint with free tier limits
        response = firecrawl_app.scrape(
            url=url,
            formats=["markdown", "html"],
            include_tags=["title", "meta", "article", "main", "content"],
            exclude_tags=["nav", "footer", "header", "aside", "script", "style"],
            timeout=10000,  # 10 seconds
            wait_for=5,  # Don't wait for JS
        )

        # Extract content from response
        result = ""
        if response and "success" in response and response["success"]:
            data = response.get("data", {})

            # Prefer markdown content, fallback to cleaned text
            if "markdown" in data and data["markdown"]:
                result = data["markdown"][:10000]
                logger.debug("Used Firecrawl markdown content")
            elif "content" in data and data["content"]:
                result = data["content"][:10000]
                logger.debug("Used Firecrawl text content")
            elif "html" in data and data["html"]:
                # Clean HTML as fallback
                soup = BeautifulSoup(data["html"], "html.parser")
                result = soup.get_text(separator=" ", strip=True)[:10000]
                logger.debug("Used Firecrawl HTML content (cleaned)")

        if result:
            logger.info(f"Firecrawl successfully scraped {url}: {len(result)} chars extracted")
        else:
            logger.warning(f"Firecrawl returned no usable content for {url}")
        return result

    except Exception as e:
        error = traceback.format_exc()
        logger.warning(f"Firecrawl failed for {url}: {error}")
        return ""


async def follow_aggregator_link(session: aiohttp.ClientSession, url: str, timeout: int = 10) -> Tuple[str, str]:
    """Follow aggregator links to find the original article URL and source."""
    logger.debug(f"Checking for aggregator links in: {url}")
    try:
        # Check if this is a known aggregator
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        if "takara.ai" in domain:
            logger.info(f"Detected Takara aggregator, attempting to find original source")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch aggregator page: HTTP {response.status}")
                    return url, domain_of(url)

                content = await response.text()
                soup = BeautifulSoup(content, "html.parser")

                # Look for common patterns where aggregators link to original sources
                original_link_selectors = [
                    'a[href*="arxiv.org"]',  # arXiv papers
                    'a[href*="github.com"]',  # GitHub links
                    'a[href*="huggingface.co"]',  # HuggingFace
                    'a[href*="doi.org"]',  # DOI links
                    'a[href*="openreview.net"]',  # OpenReview
                    'a:contains("Original")',  # Links with "Original" text
                    'a:contains("Paper")',  # Links with "Paper" text
                    'a:contains("Source")',  # Links with "Source" text
                    ".original-link",  # Common class names
                    ".source-link",
                    ".paper-link",
                ]

                for selector in original_link_selectors:
                    try:
                        link_elem = soup.select_one(selector)
                        if link_elem and link_elem.get("href"):
                            original_url = link_elem.get("href")
                            # Resolve relative URLs
                            if original_url.startswith("/"):
                                original_url = f"https://{domain}{original_url}"
                            elif not original_url.startswith("http"):
                                original_url = f"https://{domain}/{original_url}"

                            # Skip PDF files
                            if is_pdf_url(original_url):
                                logger.info(f"Skipping PDF file: {original_url}")
                                continue

                            original_domain = domain_of(original_url)
                            # Don't use the same domain as the aggregator
                            if original_domain != domain and original_domain:
                                logger.info(f"Found original source: {original_url} (from {url})")
                                return original_url, original_domain
                    except Exception as e:
                        error = traceback.format_exc()
                        logger.debug(f"Error processing selector {selector}: {error}")
                        continue

                # If no obvious link found, try to extract from meta tags or structured data
                # Look for canonical URLs or Open Graph URLs
                meta_selectors = [
                    'link[rel="canonical"]',
                    'meta[property="og:url"]',
                    'meta[name="citation_pdf_url"]',  # Academic papers
                ]

                for selector in meta_selectors:
                    try:
                        meta_elem = soup.select_one(selector)
                        if meta_elem:
                            original_url = meta_elem.get("href") or meta_elem.get("content")
                            if original_url and original_url != url:
                                # Skip PDF files
                                if is_pdf_url(original_url):
                                    logger.info(f"Skipping PDF file from meta: {original_url}")
                                    continue

                                original_domain = domain_of(original_url)
                                if original_domain != domain and original_domain:
                                    logger.info(f"Found original source via meta: {original_url} (from {url})")
                                    return original_url, original_domain
                    except Exception:
                        continue

        logger.debug(f"No aggregator pattern matched for {url}, using original")
        return url, domain_of(url)

    except Exception as e:
        error = traceback.format_exc()
        logger.warning(f"Failed to follow aggregator link {url}: {error}")
        return url, domain_of(url)


async def scrape_article_content(
    session: aiohttp.ClientSession,
    url: str,
    method: ScrapingMethod,
    firecrawl_app: Optional[FirecrawlApp] = None,
    timeout: int = 30,
) -> Tuple[str, str, str]:
    """Scrape article content and return (content, final_url, final_source)."""
    logger.debug(f"Scraping content from {url} using method {method.value}")

    # First, try to follow aggregator links to find original sources
    original_url, original_source = await follow_aggregator_link(session, url, timeout)

    # Use the original URL for scraping content
    scraping_url = original_url

    if method == ScrapingMethod.FIRECRAWL and firecrawl_app:
        content = await asyncio.get_event_loop().run_in_executor(
            None, scrape_with_firecrawl, scraping_url, firecrawl_app
        )
        if content:
            return content, original_url, original_source
        # Fallback to newspaper if Firecrawl fails
        method = ScrapingMethod.NEWSPAPER

    if method == ScrapingMethod.AUTO:
        # Try trafilatura first (modern, well-maintained), then newspaper, then beautifulsoup
        content = await scrape_with_trafilatura(session, scraping_url, timeout)
        if content:
            return content, original_url, original_source
        content = await scrape_with_newspaper(session, scraping_url, timeout)
        if content:
            return content, original_url, original_source
        content = await scrape_with_beautifulsoup(session, scraping_url, timeout)
        return content, original_url, original_source

    elif method == ScrapingMethod.TRAFILATURA:
        content = await scrape_with_trafilatura(session, scraping_url, timeout)
        if content:
            return content, original_url, original_source
        # Fallback to newspaper if trafilatura fails
        content = await scrape_with_newspaper(session, scraping_url, timeout)
        return content, original_url, original_source

    elif method == ScrapingMethod.NEWSPAPER:
        content = await scrape_with_newspaper(session, scraping_url, timeout)
        if content:
            return content, original_url, original_source
        # Fallback to beautifulsoup if newspaper fails
        content = await scrape_with_beautifulsoup(session, scraping_url, timeout)
        return content, original_url, original_source

    elif method == ScrapingMethod.BEAUTIFULSOUP:
        content = await scrape_with_beautifulsoup(session, scraping_url, timeout)
        return content, original_url, original_source

    return "", original_url, original_source
