# AI Daily Digest (uv-native)

Python-only daily AI digest. Pulls curated RSS/Atom feeds, filters last 24h, dedupes, scores for relevance, scrapes full article content, and writes a ranked digest as **Markdown and JSON**.

## Quick start (with `uv`)
```bash
# Install deps into a virtual env
uv sync

# Run with defaults (24h window, top 3 items, outputs to ./out, newspaper3k scraping)
uv run python app/digest.py

# Customise
uv run python app/digest.py --hours 24 --top 5 --format md,json,txt --out ./out

# Process only one section of feeds
uv run python app/digest.py --section "AI Research & Models"

# Choose scraping method
uv run python app/digest.py --scraper newspaper    # Default: best for articles
uv run python app/digest.py --scraper firecrawl    # Premium: requires API key  
uv run python app/digest.py --scraper beautifulsoup # Fallback: basic scraping
uv run python app/digest.py --scraper auto         # Smart: tries newspaper then beautifulsoup
```

## Web Scraping Options

### 1. Newspaper3k (Default - Recommended)
- **Best for**: News articles, blog posts, most content sites
- **Pros**: Designed specifically for article extraction, handles many edge cases
- **Cons**: May struggle with very modern JavaScript-heavy sites
- **Setup**: No additional setup required

### 2. Firecrawl (Premium Option)
- **Best for**: JavaScript-heavy sites, maximum content quality
- **Pros**: Handles dynamic content, provides markdown formatting, very clean extraction
- **Cons**: Requires API key, 500 requests/month free limit
- **Setup**: 
  ```bash
  export FIRECRAWL_API_KEY=your_api_key_here
  ```

### 3. BeautifulSoup (Fallback)
- **Best for**: Simple HTML pages, debugging
- **Pros**: Always works, no external dependencies
- **Cons**: Gets mixed content (ads, navigation), requires manual cleaning
- **Setup**: No additional setup required

### 4. Auto (Smart Fallback)
- **Best for**: Maximum reliability
- **Behavior**: Tries newspaper3k first, falls back to BeautifulSoup if it fails
- **Setup**: No additional setup required

## Features
- **RSS/Atom parsing**: Fetches from curated feeds
- **Smart web scraping**: Multiple scraping methods with intelligent fallbacks
- **Async processing**: Concurrent feed fetching and content scraping
- **Time filtering**: Configurable lookback window
- **Deduplication**: Removes duplicate articles
- **Relevance scoring**: Uses keyword matching and source weighting
- **Multiple outputs**: Markdown, JSON, and plain text formats

## Feeds
- Maintained in `app/feeds.yaml`. You can also convert an OPML file via:
```bash
uv run python app/opml_to_yaml.py --opml feeds.opml --out app/feeds.yaml
```

## GitHub Actions (daily)
A ready-to-use workflow is included at `.github/workflows/daily.yml`.
