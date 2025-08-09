#!/usr/bin/env python3
import argparse, datetime, json, re, sys, time, asyncio, os, logging
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from enum import Enum

import feedparser
from jinja2 import Environment, FileSystemLoader, select_autoescape
import yaml
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from newspaper import Article
from firecrawl import FirecrawlApp

# Set up logging
logger = logging.getLogger(__name__)

class ScrapingMethod(Enum):
    NEWSPAPER = "newspaper"
    BEAUTIFULSOUP = "beautifulsoup" 
    FIRECRAWL = "firecrawl"
    AUTO = "auto"

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

def domain_of(link: str) -> str:
    try:
        domain = urlparse(link).netloc.replace("www.","")
        logger.debug(f"Extracted domain: {domain} from {link}")
        return domain
    except Exception as e:
        logger.warning(f"Failed to extract domain from {link}: {e}")
        return ""

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

def score_item(item: Dict[str, Any], kw: Dict[str, Any]) -> int:
    text = f"{item['title']} {item.get('summary','')} {item.get('content','')}".lower()
    s = 0
    
    must_matches = []
    for k in kw.get("must", []):
        if k and k.lower() in text:
            s += 2
            must_matches.append(k)
    
    nice_matches = []
    for k in kw.get("nice", []):
        if k and k.lower() in text:
            s += 1
            nice_matches.append(k)
    
    src = item.get("source","")
    source_adjustments = []
    if any(d in src for d in kw.get("source_weight", {}).get("plus", [])):
        s += 1
        source_adjustments.append("+1 (source bonus)")
    if any(d in src for d in kw.get("source_weight", {}).get("minus", [])):
        s -= 1
        source_adjustments.append("-1 (source penalty)")
    
    logger.debug(f"Scoring item '{item['title'][:50]}...': score={s}, "
                f"must_matches={must_matches}, nice_matches={nice_matches}, "
                f"source_adjustments={source_adjustments}")
    return s

def extract_content(entry) -> str:
    """Extract content from feed entry, preferring content over summary."""
    content = ""
    
    # Try to get content from various possible fields
    if hasattr(entry, 'content') and entry.content:
        if isinstance(entry.content, list) and len(entry.content) > 0:
            content = entry.content[0].get('value', '')
        else:
            content = str(entry.content)
    elif hasattr(entry, 'description') and entry.description:
        content = entry.description
    elif hasattr(entry, 'summary') and entry.summary:
        content = entry.summary
    
    return clean_html(content)

async def scrape_with_newspaper(session: aiohttp.ClientSession, url: str, timeout: int = 10) -> str:
    """Scrape content using newspaper3k library."""
    logger.debug(f"Starting newspaper scraping for: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            response.raise_for_status()
            html_content = await response.text()
        
        logger.debug(f"Downloaded HTML content: {len(html_content)} chars")
        
        # Use newspaper3k to parse the article
        article = Article(url, language='en')
        article.set_html(html_content)
        article.parse()
        
        # Get the article text
        text = article.text.strip()
        
        # Add title if available and not already in text
        if article.title and article.title not in text[:200]:
            text = f"{article.title}\n\n{text}"
        
        result = text[:3000]  # Limit content size
        logger.info(f"Newspaper3k successfully scraped {url}: {len(result)} chars extracted")
        return result
        
    except Exception as e:
        logger.warning(f"Newspaper3k failed for {url}: {e}")
        return ""

async def scrape_with_beautifulsoup(session: aiohttp.ClientSession, url: str, timeout: int = 10) -> str:
    """Scrape content using BeautifulSoup (fallback method)."""
    logger.debug(f"Starting BeautifulSoup scraping for: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            response.raise_for_status()
            content = await response.text()
        
        logger.debug(f"Downloaded HTML content: {len(content)} chars")
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Try to find main content areas (common article selectors)
        content_selectors = [
            'article',
            '[role="main"]',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            'main',
            '.main-content'
        ]
        
        content_text = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content_text = content_elem.get_text(separator=' ', strip=True)
                break
        
        # Fallback to body if no main content found
        if not content_text:
            content_text = soup.body.get_text(separator=' ', strip=True) if soup.body else ""
        
        # Clean up whitespace and limit size
        content_text = re.sub(r'\s+', ' ', content_text).strip()
        result = content_text[:3000]
        logger.info(f"BeautifulSoup successfully scraped {url}: {len(result)} chars extracted")
        return result
        
    except Exception as e:
        logger.warning(f"BeautifulSoup failed for {url}: {e}")
        return ""

async def scrape_with_firecrawl(url: str, firecrawl_app: Optional[FirecrawlApp] = None) -> str:
    """Scrape content using Firecrawl API."""
    if not firecrawl_app:
        logger.debug("No Firecrawl app provided, returning empty string")
        return ""
    
    logger.debug(f"Starting Firecrawl scraping for: {url}")
    try:
        # Use Firecrawl's scrape endpoint with free tier limits
        response = firecrawl_app.scrape_url(
            url=url,
            params={
                'formats': ['markdown', 'html'],
                'includeTags': ['title', 'meta', 'article', 'main', 'content'],
                'excludeTags': ['nav', 'footer', 'header', 'aside', 'script', 'style'],
                'timeout': 10000,  # 10 seconds
                'waitFor': 0,      # Don't wait for JS
            }
        )
        
        # Extract content from response
        result = ""
        if response and 'success' in response and response['success']:
            data = response.get('data', {})
            
            # Prefer markdown content, fallback to cleaned text
            if 'markdown' in data and data['markdown']:
                result = data['markdown'][:3000]
                logger.debug("Used Firecrawl markdown content")
            elif 'content' in data and data['content']:
                result = data['content'][:3000]
                logger.debug("Used Firecrawl text content")
            elif 'html' in data and data['html']:
                # Clean HTML as fallback
                soup = BeautifulSoup(data['html'], 'html.parser')
                result = soup.get_text(separator=' ', strip=True)[:3000]
                logger.debug("Used Firecrawl HTML content (cleaned)")
        
        if result:
            logger.info(f"Firecrawl successfully scraped {url}: {len(result)} chars extracted")
        else:
            logger.warning(f"Firecrawl returned no usable content for {url}")
        return result
        
    except Exception as e:
        logger.warning(f"Firecrawl failed for {url}: {e}")
        return ""

def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file."""
    try:
        parsed = urlparse(url.lower())
        path = parsed.path.lower()
        is_pdf = (path.endswith('.pdf') or 
                 'pdf' in parsed.query.lower() or
                 'content-type=application/pdf' in parsed.query.lower())
        logger.debug(f"PDF check for {url}: {is_pdf}")
        return is_pdf
    except Exception as e:
        logger.warning(f"Error checking if URL is PDF {url}: {e}")
        return False

async def follow_aggregator_link(session: aiohttp.ClientSession, url: str, timeout: int = 10) -> tuple[str, str]:
    """Follow aggregator links to find the original article URL and source."""
    logger.debug(f"Checking for aggregator links in: {url}")
    try:
        # Check if this is a known aggregator
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        if "takara.ai" in domain:
            logger.info(f"Detected Takara aggregator, attempting to find original source")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch aggregator page: HTTP {response.status}")
                    return url, domain_of(url)
                    
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
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
                    '.original-link',  # Common class names
                    '.source-link',
                    '.paper-link'
                ]
                
                for selector in original_link_selectors:
                    try:
                        link_elem = soup.select_one(selector)
                        if link_elem and link_elem.get('href'):
                            original_url = link_elem.get('href')
                            # Resolve relative URLs
                            if original_url.startswith('/'):
                                original_url = f"https://{domain}{original_url}"
                            elif not original_url.startswith('http'):
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
                        logger.debug(f"Error processing selector {selector}: {e}")
                        continue
                
                # If no obvious link found, try to extract from meta tags or structured data
                # Look for canonical URLs or Open Graph URLs
                meta_selectors = [
                    'link[rel="canonical"]',
                    'meta[property="og:url"]',
                    'meta[name="citation_pdf_url"]'  # Academic papers
                ]
                
                for selector in meta_selectors:
                    try:
                        meta_elem = soup.select_one(selector)
                        if meta_elem:
                            original_url = meta_elem.get('href') or meta_elem.get('content')
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
        logger.warning(f"Failed to follow aggregator link {url}: {e}")
        return url, domain_of(url)

async def scrape_article_content(session: aiohttp.ClientSession, url: str, method: ScrapingMethod, firecrawl_app: Optional[FirecrawlApp] = None, timeout: int = 10) -> tuple[str, str, str]:
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
        # Try newspaper first, then beautifulsoup
        content = await scrape_with_newspaper(session, scraping_url, timeout)
        if content:
            return content, original_url, original_source
        content = await scrape_with_beautifulsoup(session, scraping_url, timeout)
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

async def fetch_feed_items(session: aiohttp.ClientSession, group: str, url: str, hours: int, scraping_method: ScrapingMethod, firecrawl_app: Optional[FirecrawlApp] = None) -> List[Dict[str, Any]]:
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
            
            items.append({
                "title": (e.get("title") or "").strip(),
                "link": link,
                "content": feed_content,
                "full_content": scraped_content,
                "source": source,
                "group": group,
                "date": extract_date(e),
            })
        
        logger.info(f"Successfully processed {len(items)} items from {group} "
                   f"({success_count} with scraped content)")
        
    except Exception as e:
        logger.error(f"Failed to fetch feed {url}: {e}")
    
    return items

def has_sufficient_content(item: Dict[str, Any], min_length: int = 300) -> bool:
    """Check if item has sufficient content in either content or full_content field."""
    content = item.get("content", "") or ""
    full_content = item.get("full_content", "") or ""
    
    content_length = len(content.strip())
    full_content_length = len(full_content.strip())
    
    has_sufficient = content_length >= min_length or full_content_length >= min_length
    
    logger.debug(f"Content length check for '{item.get('title', 'Unknown')[:50]}...': "
                f"content={content_length}, full_content={full_content_length}, "
                f"min_required={min_length}, sufficient={has_sufficient}")
    
    return has_sufficient

async def fetch_all(feeds: Dict[str, list], hours: int, scraping_method: ScrapingMethod, section: Optional[str] = None) -> List[Dict[str, Any]]:
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
        api_key = os.getenv('FIRECRAWL_API_KEY')
        if api_key:
            try:
                firecrawl_app = FirecrawlApp(api_key=api_key)
                logger.info("Using Firecrawl for web scraping")
            except Exception as e:
                logger.warning(f"Failed to initialize Firecrawl: {e}")
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
    
    # Filter items with insufficient content
    content_filtered = []
    insufficient_content = 0
    for item in deduped:
        if has_sufficient_content(item):
            content_filtered.append(item)
        else:
            insufficient_content += 1
    
    logger.info(f"Content filtering complete: {len(content_filtered)} items with sufficient content, "
               f"{insufficient_content} items filtered out for insufficient content")
    
    return content_filtered

async def render_outputs(items: List[Dict[str, Any]], out_dir: str, formats: list) -> None:
    logger.info(f"Rendering outputs to {out_dir} in formats: {formats}")
    
    out_dir = out_dir.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
    base = f"{out_dir}/digest-{ts}"
    env = Environment(loader=FileSystemLoader(searchpath="app/templates"),
                      autoescape=select_autoescape())
    
    # Prepare all outputs
    output_tasks = []
    
    if "json" in formats:
        json_content = json.dumps(items, indent=2, ensure_ascii=False)
        output_tasks.append(write_file(base + ".json", json_content))
    
    if "md" in formats:
        tpl = env.get_template("markdown.md.j2")
        md = tpl.render(items=items, ts=ts)
        output_tasks.append(write_file(base + ".md", md))
    
    if "txt" in formats:
        tpl = env.get_template("text.txt.j2")
        txt = tpl.render(items=items, ts=ts)
        output_tasks.append(write_file(base + ".txt", txt))
    
    # Write all files concurrently
    if output_tasks:
        await asyncio.gather(*output_tasks)
        logger.info(f"Successfully wrote {len(output_tasks)} output files")
    else:
        logger.warning("No output formats specified")

async def write_file(filepath: str, content: str) -> None:
    """Write content to file asynchronously."""
    try:
        async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
            await f.write(content)
        logger.debug(f"Successfully wrote file: {filepath}")
    except Exception as e:
        logger.error(f"Failed to write file {filepath}: {e}")
        raise

async def main():
    p = argparse.ArgumentParser(description="AI Daily Digest (async)")
    p.add_argument("--feeds", default="app/feeds.yaml", help="YAML file with feed groups")
    p.add_argument("--keywords", default="app/keywords.json", help="JSON file with keyword scoring")
    p.add_argument("--hours", type=int, default=24, help="lookback window in hours")
    p.add_argument("--top", type=int, default=3, help="top N items to keep")
    p.add_argument("--out", default="./out", help="output directory")
    p.add_argument("--format", default="json", help="comma-separated: md,json,txt")
    p.add_argument("--section", help="process only one section/group from feeds (e.g., 'AI Research & Models')")
    p.add_argument("--scraper", choices=['newspaper', 'beautifulsoup', 'firecrawl', 'auto'], 
                   default='newspaper', help="Web scraping method (default: newspaper)")
    p.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                   default='INFO', help="Set the logging level (default: INFO)")
    args = p.parse_args()

    # Setup logging with the specified level
    setup_logging(args.log_level)
    
    logger.info("Starting AI Daily Digest")
    logger.debug(f"Arguments: {vars(args)}")

    try:
        with open(args.feeds, "r", encoding="utf-8") as f:
            feeds = yaml.safe_load(f)
        logger.info(f"Loaded {len(feeds)} feed groups from {args.feeds}")
        
        with open(args.keywords, "r", encoding="utf-8") as f:
            kw = json.load(f)
        logger.info(f"Loaded keywords configuration from {args.keywords}")
    except Exception as e:
        logger.error(f"Failed to load configuration files: {e}")
        return 1

    scraping_method = ScrapingMethod(args.scraper)
    items = await fetch_all(feeds, hours=args.hours, scraping_method=scraping_method, section=args.section)
    
    logger.info("Starting item scoring...")
    for it in items:
        it["_score"] = score_item(it, kw)
    
    items.sort(key=lambda x: x["_score"], reverse=True)
    top = items[: args.top]
    
    logger.info(f"Selected top {len(top)} items (scores: {[item['_score'] for item in top]})")

    fmts = [x.strip() for x in args.format.split(",") if x.strip()]
    await render_outputs(top, args.out, fmts)
    
    logger.info(f"Digest generation complete! Wrote digest with {len(top)} items to {args.out}")
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
