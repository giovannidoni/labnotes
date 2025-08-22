# Labnotes - AI Daily Digest

Python-based AI digest system that fetches RSS/Atom feeds, filters articles, scores for relevance, scrapes full content, and outputs ranked digests. Includes smart deduplication using embeddings to remove similar articles.

## Installation

```bash
# Install the package in editable mode
pip install -e .

# This makes the labnotes command available system-wide
```

## Quick Start

### Generate a Digest

```bash
# Basic usage - generates digest with default settings (24h lookback, top 3 items)
labnotes

# Customize parameters
labnotes --hours 48 --top 5 --format md,json,txt --out ./output

# Process only one section of feeds
labnotes --section "AI Research & Models"

# Use different embedding models for scoring
labnotes --model text-embedding-3-small  # OpenAI (requires OPENAI_API_KEY)
labnotes --model all-MiniLM-L6-v2        # Local sentence-transformers (default)
```

### Deduplicate Similar Articles

After generating a digest, you can remove similar articles using embedding-based similarity:

```bash
# Basic deduplication (keeps most recent of similar articles)
# Processes the most recent digest file in the specified section directory
labnotes-dedup --input /path/to/digest/output --section ai_research_and_models

# Keep highest scoring articles instead of most recent
labnotes-dedup --input /path/to/digest/output --section tech_news --prefer-score

# Adjust similarity threshold (0-1, higher = more strict)
labnotes-dedup --input /path/to/digest/output --section ai_research_and_models --threshold 0.9

# Use OpenAI embeddings for better similarity detection
labnotes-dedup --input /path/to/digest/output --section ai_research_and_models --model text-embedding-3-small
```

**Output Enhancement**: The deduplication process now automatically saves embeddings as an additional field in the output JSON. Each deduplicated item will include an `embedding` field containing the vector representation used for similarity detection, enabling further analysis or custom similarity comparisons.

## Commands

- **`labnotes`** - Generate digest from RSS/Atom feeds
- **`labnotes-dedup`** - Remove similar articles from digest output

## Configuration Files

### Feeds Configuration
Feeds are configured in `labnotes/data/feeds.yaml`:
```yaml
AI Research & Models:
  - https://arxiv.org/rss/cs.AI
  - https://arxiv.org/rss/cs.LG
  
Tech News:
  - https://feeds.feedburner.com/TechCrunch
  - https://rss.cnn.com/rss/edition.rss
```

Convert OPML to YAML:
```bash
python -m labnotes.opml_to_yaml feeds.opml -o labnotes/data/feeds.yaml
```

### Keywords Configuration
Scoring keywords in `labnotes/settings/website/keywords.json`:
```json
{
  "must": ["AI", "machine learning", "LLM"],
  "nice": ["research", "breakthrough", "model"],
  "source_weight": {
    "plus": ["arxiv.org", "nature.com"],
    "minus": ["clickbait.com"]
  }
}
```

## Web Scraping Options

The system supports multiple scraping methods that are automatically selected:

### 1. Newspaper3k (Default)
- **Best for**: News articles, blog posts, most content sites
- **Pros**: Designed for article extraction, handles many edge cases
- **Setup**: No additional setup required

### 2. OpenAI Embeddings
- **Best for**: High-quality similarity detection and deduplication
- **Models**: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- **Setup**: Set `OPENAI_API_KEY` environment variable
- **Usage**: Automatically detected by model name

### 3. Firecrawl (Premium)
- **Best for**: JavaScript-heavy sites, maximum content quality
- **Setup**: Set `FIRECRAWL_API_KEY` environment variable
- **Usage**: Add `--scraper firecrawl` to labnotes command

### 4. Sentence Transformers (Local)
- **Best for**: Privacy-focused embedding generation
- **Models**: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, etc.
- **Setup**: No additional setup required (downloads models automatically)

## Features

### Core Functionality
- **RSS/Atom parsing**: Fetches from curated feeds with async processing
- **Smart web scraping**: Multiple scraping methods with intelligent fallbacks
- **Time filtering**: Configurable lookback window (hours, days)
- **Content quality filtering**: Removes low-quality content automatically
- **Relevance scoring**: Keyword matching and source weighting
- **Multiple outputs**: Markdown, JSON, and plain text formats

### Advanced Features
- **Embedding-based deduplication**: Remove similar articles using AI embeddings
- **Provider auto-detection**: Automatically chooses OpenAI vs local embeddings based on model name
- **Section-based processing**: Process specific feed categories
- **Concurrent processing**: Async fetching and content scraping for speed

## Environment Variables

```bash
# Optional: For OpenAI embeddings (better similarity detection)
export OPENAI_API_KEY=your_openai_api_key

# Optional: For premium web scraping
export FIRECRAWL_API_KEY=your_firecrawl_api_key
```

## Examples

### Complete Workflow

```bash
# 1. Generate initial digest
labnotes --hours 24 --top 10 --format json --out ./output

# 2. Remove similar articles
labnotes-dedup --input ./output --section ai_research_and_models --threshold 0.85

# 3. Generate final markdown output
labnotes --hours 24 --top 5 --format md --out ./final
```

### Advanced Usage

```bash
# High-quality processing with OpenAI
export OPENAI_API_KEY=your_key
labnotes --model text-embedding-3-small --scraper firecrawl
labnotes-dedup --input ./output --section ai_research_and_models --model text-embedding-3-small --threshold 0.9

# Privacy-focused local processing
labnotes --model all-MiniLM-L6-v2
labnotes-dedup --input ./output --section ai_research_and_models --model all-mpnet-base-v2
```

## File Structure

```
labnotes/
├── __init__.py
├── digest.py          # Main digest generation
├── deduplicate.py     # Similarity-based deduplication
├── embeddings.py      # Embedding generation (OpenAI + local)
├── similarity.py      # Similarity detection algorithms
├── scraping.py        # Web content scraping
├── utils.py           # Utilities and logging
├── opml_to_yaml.py    # OPML conversion utility
├── data/
│   ├── feeds.yaml     # RSS/Atom feed configuration
│   └── keywords.json  # Scoring keywords and weights
└── templates/
    ├── markdown.md.j2 # Markdown output template
    └── text.txt.j2    # Plain text output template
```

## Development

```bash
# Install in development mode
pip install -e .

# Run with debug logging
labnotes --log-level DEBUG
labnotes-dedup --input ./output --section ai_research_and_models --log-level DEBUG
```