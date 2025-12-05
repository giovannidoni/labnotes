import logging
import random
import time
import traceback
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# from ladnotes.llm.openai_utils import query_llm_sync
from labnotes.scraping_tools.escape_cookie_banner import kill_cookie_banners, wait_for_page_content

try:
    from bs4 import BeautifulSoup

    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


logger = logging.getLogger(__name__)



def get_driver(headless: bool = True):
    """ Get a Selenium WebDriver instance with stealth settings."""
    if "driver" in locals():
        driver.quit()
    return make_driver(headless=headless)


def get_links(url: str, driver=None, apply_filter=None):
    """ Get all links from a webpage using Selenium."""
    if driver is None:
        driver = get_driver(headless=True)
    driver.get(url)
    links = driver.find_elements(By.TAG_NAME, "a")
    return extract_links(links, apply_filter=apply_filter)


def extract_links(links: list, apply_filter: str):
    track_href = set()
    links_out = set()
    for link in links:
        href = link.get_attribute("href")
        link_text = link.text.strip()
        if href and href not in track_href:
            track_href.add(href)
            if apply_filter:
                if apply_filter in str(href):
                    links_out.add((href, link_text))
            else:
                links_out.add((href, link_text))
    return list(links_out)


def _domain_only(value: str) -> str:
    p = urlparse(value if "://" in str(value) else f"https://{value}")
    return (p.hostname or "").lstrip("www.")


def _is_url(text: str) -> bool:
    if not isinstance(text, str):
        return False
    if "www" in text or "https" in text:
        try:
            result = urlparse(text)
            return result.scheme in ("http", "https") and bool(result.netloc)
        except Exception:
            return False
    return False


def get_clean_page_text(driver, url: str | None = None) -> str:
    # Wait for main content to load (see previous wait_for_page_content function)
    if url:
        driver.get(url)
    wait_for_page_content(driver, timeout=1)

    # Get fully rendered HTML
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag in soup(["style", "script", "noscript"]):
        tag.decompose()

    # Prefer <main> or <article> if present
    main = soup.find("main") or soup.find("article") or soup
    clean_text = main.get_text(separator="\n", strip=True)
    return clean_text


def scrape_website(
    start_url: str,
    driver,
    *,
    max_pages: int = 10,
    delay_range: tuple = (1, 3),
    include_links: bool = True,
    claude_args: dict | None,
) -> dict:
    """
    Scrape a website starting from start_url and following internal links.

    Args:
        start_url: The initial URL to start scraping from
        driver: Selenium WebDriver instance
        max_pages: Maximum number of pages to scrape
        delay_range: Tuple of (min, max) seconds to wait between requests
        include_links: Whether to include the list of found links in the returned data

    Returns:
        Dict with URL as key and page data as value
    """
    from urllib.parse import urljoin, urlparse

    base_domain = _domain_only(start_url)
    visited_urls = set()
    scraped_data = {}
    urls_to_visit = [start_url]
    max_pages = max_pages if include_links else 1

    logger.info("Starting scrape of %s (max %d pages)", start_url, max_pages)

    while urls_to_visit and len(scraped_data) < max_pages:
        if current_url := urls_to_visit.pop(0):
            logger.info("Queue has %d URLs remaining to visit", len(urls_to_visit))

            if current_url in visited_urls:
                continue

            if not _is_url(current_url):
                logger.warning("Skipping invalid URL: %s", current_url)
                continue

            try:
                # Navigate to the page
                logger.info("Scraping page %d/%d: %s", len(scraped_data) + 1, max_pages, current_url)
                driver.get(current_url)
                kill_cookie_banners(driver, timeout=2)
                visited_urls.add(current_url)

                # # Human-like delay
                # time.sleep(random.uniform(*delay_range))

                # Prettify HTML if requested and BeautifulSoup is available
                page = get_clean_page_text(driver)

                # Get page data
                page_data = {
                    "title": driver.title,
                    "text": page,
                }
                if claude_args["prompt"] and len(page) > 500:
                    logging.info(f"Sending {current_url} page to Claude for analysis...")
                    tmp = [{"role": "user", "content": claude_args["prompt"] + page}]
                    response_format = {
                        "type": "json_schema",
                        "json_schema": {"name": "article_summary", "schema": claude_args["schema"], "strict": True},
                    }
                    res = query_llm_sync(
                        tmp,
                        model=claude_args["model"],
                        max_tokens=claude_args["max_tokens"],
                        response_format=response_format,
                    )

                    page_data["cleaned_metadata"] = res["cleaned_metadata"]
                    page_data["relevant_page"] = res["relevant_page"]

                # Find all internal links
                links = driver.find_elements(By.TAG_NAME, "a")
                for href, link_text in extract_links(links):
                    try:
                        # Skip mailto and other non-http protocols
                        keywords = (
                            "login",
                            "sign-in",
                            "register",
                            "signin",
                            "contact",
                            "signup",
                            "cookies",
                            "legal",
                            "privacy",
                            "terms",
                            "policy",
                        )
                        if any(keyword in href.lower() for keyword in keywords):
                            logger.info("Skipping link: %s", href)
                            continue
                        if href.startswith(("mailto:", "tel:", "javascript:", "ftp:")):
                            continue

                        # Convert relative URLs to absolute
                        full_url = urljoin(current_url, href)
                        link_domain = _domain_only(full_url)
                        links_found = []

                        # Only include links from the same domain (filter out outbound links)
                        if link_domain == base_domain and full_url not in visited_urls:
                            # Check if it's likely an HTML page (not file download)
                            if not any(
                                full_url.lower().endswith(ext)
                                for ext in [".pdf", ".doc", ".docx", ".zip", ".jpg", ".png", ".gif", ".css", ".js"]
                            ):
                                if include_links:
                                    # Store link with same structure as page_data
                                    links_found.append(full_url)

                                if full_url not in urls_to_visit:
                                    urls_to_visit.append(full_url)
                    except Exception as e:
                        logger.debug("Error processing link: %s", e)
                        continue

                scraped_data[current_url] = page_data
                links_count = len(page_data.get("links_found", {}))
                logger.info(
                    "Scraped: %s (%d chars, %d internal links found)", current_url, len(page_data["text"]), links_count
                )
                if include_links:
                    logger.info(
                        "Added %d new URLs to queue (total queue size: %d)",
                        len([url for url in links_found if url not in visited_urls]),
                        len(urls_to_visit),
                    )

            except Exception as e:
                error = traceback.format_exc()
                logger.error("Error scraping %s: %s", current_url, error)
                continue
        else:
            page_data = {
                "url": start_url,
                "title": "",
                # 'html': formatted_html,
                "text": "",
                "links_found": {},
            }

    logger.info("Scraping complete. Scraped %d pages total", len(scraped_data))
    return scraped_data


def make_driver(headless: bool = False):  # Changed default to False for better stealth
    opts = Options()

    # Stealth options to avoid detection
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-browser-side-navigation")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--log-level=3")

    # Add a realistic user agent
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    if headless:
        opts.add_argument("--headless=new")

    # Remove automation indicators
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)

    # Execute script to remove webdriver property
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    return driver


def find_pagination(driver):
    """Find pagination elements using common selectors."""
    pagination_selectors = [
        # Common class names
        (By.CLASS_NAME, "pagination"),
        (By.CLASS_NAME, "pager"),
        (By.CLASS_NAME, "page-numbers"),
        (By.CLASS_NAME, "wp-pagenavi"),  # WordPress
        
        # Common IDs
        (By.ID, "pagination"),
        (By.ID, "pager"),
        
        # ARIA labels
        (By.CSS_SELECTOR, "[aria-label*='pagination' i]"),
        (By.CSS_SELECTOR, "[aria-label*='pager' i]"),
        
        # Nav elements with pagination role
        (By.CSS_SELECTOR, "nav[role='navigation']"),
    ]
    
    for by, selector in pagination_selectors:
        try:
            elements = driver.find_elements(by, selector)
            if elements:
                return elements
        except:
            continue
    
    return []


def find_page_links(driver):
    """Find all pagination page number links."""
    page_links = []
    
    # Try to find pagination container first
    pagination = find_pagination(driver)
    unique_links = set()
    
    if pagination:
        # Look for links within pagination
        links = pagination[0].find_elements(By.TAG_NAME, "a")
        for link in links:
            href = link.get_attribute("href")
            text = link.text.strip()
            
            # Filter out non-numeric links (Next, Previous, etc.)
            if (text.isdigit() or href) and href not in unique_links:
                unique_links.add(href)
                page_links.append({
                    'url': href,
                    'text': text,
                    'element': link
                })
    else:
        # Fallback: look for links that look like page numbers
        all_links = driver.find_elements(By.TAG_NAME, "a")
        for link in all_links:
            text = link.text.strip()
            href = link.get_attribute("href")
            
            # Check if it looks like a page number
            if text.isdigit() and 1 <= int(text) <= 1000:
                page_links.append({
                    'url': href,
                    'text': text,
                    'element': link
                })
    
    return page_links