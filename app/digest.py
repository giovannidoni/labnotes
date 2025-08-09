#!/usr/bin/env python3
import argparse, datetime, json, re, sys, time, asyncio
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

import feedparser
from jinja2 import Environment, FileSystemLoader, select_autoescape
import yaml
import aiohttp
import aiofiles
from bs4 import BeautifulSoup

def domain_of(link: str) -> str:
    try:
        return urlparse(link).netloc.replace("www.","")
    except Exception:
        return ""

def clean_html(s: str) -> str:
    return re.sub("<[^<]+?>", "", s or "")

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
    return (dt is not None) and (dt >= cutoff)

def score_item(item: Dict[str, Any], kw: Dict[str, Any]) -> int:
    text = f"{item['title']} {item.get('summary','')} {item.get('content','')}".lower()
    s = 0
    for k in kw.get("must", []):
        if k and k.lower() in text:
            s += 2
    for k in kw.get("nice", []):
        if k and k.lower() in text:
            s += 1
    src = item.get("source","")
    if any(d in src for d in kw.get("source_weight", {}).get("plus", [])):
        s += 1
    if any(d in src for d in kw.get("source_weight", {}).get("minus", [])):
        s -= 1
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

async def scrape_article_content(session: aiohttp.ClientSession, url: str, timeout: int = 10) -> str:
    """Scrape the full article content from a URL asynchronously."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            response.raise_for_status()
            content = await response.text()
        
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
        
        # Clean up whitespace
        content_text = re.sub(r'\s+', ' ', content_text).strip()
        
        return content_text
        
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return ""

async def fetch_feed_items(session: aiohttp.ClientSession, group: str, url: str, hours: int) -> List[Dict[str, Any]]:
    """Fetch items from a single feed asynchronously."""
    items = []
    try:
        # Fetch feed content
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            feed_content = await response.text()
        
        d = feedparser.parse(feed_content)
        
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
            scraping_tasks.append(scrape_article_content(session, link))
        
        if not valid_entries:
            return items
        
        print(f"Scraping {len(valid_entries)} items from {group}...")
        scraped_contents = await asyncio.gather(*scraping_tasks, return_exceptions=True)
        
        for e, scraped_content in zip(valid_entries, scraped_contents):
            if isinstance(scraped_content, Exception):
                scraped_content = ""
            
            link = e.get("link") or e.get("id") or ""
            
            # Extract and clean feed content (remove HTML tags)
            raw_feed_content = extract_content(e)
            feed_content = clean_html(raw_feed_content)
            
            items.append({
                "title": (e.get("title") or "").strip(),
                "link": link,
                "content": feed_content,
                "full_content": scraped_content,
                "source": domain_of(link),
                "group": group,
                "date": extract_date(e),
            })
    except Exception as e:
        print(f"Failed to fetch feed {url}: {e}")
    
    return items

async def fetch_all(feeds: Dict[str, list], hours: int, section: Optional[str] = None) -> List[Dict[str, Any]]:
    # Filter feeds to only the specified section if provided
    if section:
        if section not in feeds:
            print(f"Warning: Section '{section}' not found in feeds. Available sections: {list(feeds.keys())}")
            return []
        feeds = {section: feeds[section]}
    
    async with aiohttp.ClientSession() as session:
        # Create tasks for all feeds
        feed_tasks = []
        for group, urls in feeds.items():
            for url in urls:
                feed_tasks.append(fetch_feed_items(session, group, url, hours))
        
        # Fetch all feeds concurrently
        print(f"Fetching {len(feed_tasks)} feeds concurrently...")
        feed_results = await asyncio.gather(*feed_tasks, return_exceptions=True)
        
        # Flatten results
        items = []
        for result in feed_results:
            if isinstance(result, Exception):
                continue
            items.extend(result)
    
    # Dedupe by link
    seen = set()
    deduped = []
    for it in items:
        if it["link"] in seen:
            continue
        seen.add(it["link"])
        deduped.append(it)
    return deduped

async def render_outputs(items: List[Dict[str, Any]], out_dir: str, formats: list) -> None:
    import os
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

async def write_file(filepath: str, content: str) -> None:
    """Write content to file asynchronously."""
    async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
        await f.write(content)

async def main():
    import os
    p = argparse.ArgumentParser(description="AI Daily Digest (async)")
    p.add_argument("--feeds", default="app/feeds.yaml", help="YAML file with feed groups")
    p.add_argument("--keywords", default="app/keywords.json", help="JSON file with keyword scoring")
    p.add_argument("--hours", type=int, default=24, help="lookback window in hours")
    p.add_argument("--top", type=int, default=3, help="top N items to keep")
    p.add_argument("--out", default="./out", help="output directory")
    p.add_argument("--format", default="json", help="comma-separated: md,json,txt")
    p.add_argument("--section", help="process only one section/group from feeds (e.g., 'AI Research & Models')")
    args = p.parse_args()

    with open(args.feeds, "r", encoding="utf-8") as f:
        feeds = yaml.safe_load(f)
    with open(args.keywords, "r", encoding="utf-8") as f:
        kw = json.load(f)

    items = await fetch_all(feeds, hours=args.hours, section=args.section)
    for it in items:
        it["_score"] = score_item(it, kw)
    items.sort(key=lambda x: x["_score"], reverse=True)
    top = items[: args.top]

    fmts = [x.strip() for x in args.format.split(",") if x.strip()]
    await render_outputs(top, args.out, fmts)
    print(f"Wrote digest with {len(top)} items to {args.out}")

if __name__ == "__main__":
    asyncio.run(main())
