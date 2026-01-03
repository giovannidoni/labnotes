# Labnotes - AI Daily Digest

Python-based AI digest system that fetches RSS/Atom feeds, filters articles, scores for relevance, scrapes full content, and outputs ranked digests. Includes smart deduplication using embeddings and AI-powered summarization.

## Installation

```bash
# Install with uv
uv sync

# Or install the package in editable mode
pip install -e .
```

## Quick Start

### Generate a Digest

```bash
# Basic usage - generates digest with settings from settings.yaml
labnotes --section ai_research_and_models --out ./out

# Use different scraper
labnotes --section ai_research_and_models --scraper firecrawl --out ./out
```

### Deduplicate Similar Articles

```bash
# Remove similar articles using embedding-based similarity
labnotes-dedup --input ./out --section ai_research_and_models

# Adjust similarity threshold (0-1, higher = more strict)
labnotes-dedup --input ./out --section ai_research_and_models --threshold 0.9

# Keep highest scoring articles instead of most recent
labnotes-dedup --input ./out --section ai_research_and_models --prefer-score
```

### Summarize Articles

```bash
# Generate AI summaries for deduplicated articles
labnotes-summarise --input ./out --section ai_research_and_models
```

### Collect Results

```bash
# Collect and summarize results across all sections
labnotes-collect --input ./out
```

## Commands

- **`labnotes`** - Generate digest from RSS/Atom feeds
- **`labnotes-dedup`** - Remove similar articles using embeddings
- **`labnotes-summarise`** - Add AI summaries to articles
- **`labnotes-collect`** - Collect and summarize results across sections
- **`labnotes-post`** - Publish results to Slack/LinkedIn

## Configuration

All settings are centralized in `labnotes/settings/settings.yaml`:

```yaml
website:
  llm_model: "gemini/gemini-2.0-flash"      # LLM for summarization
  embedding_model: "gemini/text-embedding-004"  # Embeddings for deduplication
  digest:
    top: 100
    hours: 168
  summarise:
    model: "@format {this.llm_model}"  # Inherits from top-level
    max_concurrent: 5
  dedup:
    model: "@format {this.embedding_model}"  # Inherits from top-level
  collect:
    model: "@format {this.llm_model}"
    min_score: 8.0
    min_novelty_score: 2.0
    filter_mode: "AND"
```

### Environment Selection

Use `ENV_FOR_DYNACONF` to select configuration:

```bash
ENV_FOR_DYNACONF=website labnotes --section ai_research_and_models
ENV_FOR_DYNACONF=website-local labnotes --section ai_research_and_models
```

### Feeds Configuration

Feeds are configured in `labnotes/data/feeds.yaml`:

```yaml
ai_research_and_models:
  - https://arxiv.org/rss/cs.AI
  - https://arxiv.org/rss/cs.LG

community_signals:
  - https://hnrss.org/frontpage
```

### Keywords Configuration

Scoring keywords in `labnotes/settings/website/keywords.json`:

```json
{
  "audiences": {
    "manager": {
      "must": ["AI", "strategy"],
      "nice": ["research", "breakthrough"]
    }
  }
}
```

## Web Scraping Options

```bash
# Newspaper3k (default) - best for news articles
labnotes --section ai_research_and_models --scraper newspaper

# BeautifulSoup - lightweight HTML parsing
labnotes --section ai_research_and_models --scraper beautifulsoup

# Firecrawl - JavaScript-heavy sites (requires FIRECRAWL_API_KEY)
labnotes --section ai_research_and_models --scraper firecrawl
```

## Environment Variables

```bash
# Required: For Gemini models
export GEMINI_API_KEY=your_gemini_api_key

# Optional: For Firecrawl scraping
export FIRECRAWL_API_KEY=your_firecrawl_api_key

# Optional: For Supabase storage
export SUPABASE_URL=your_supabase_url
export SUPABASE_TOKEN=your_supabase_token

# Optional: For publishing
export SLACK_BOT_TOKEN=your_slack_token
export LINKEDIN_API_TOKEN=your_linkedin_token
```

## Complete Workflow

```bash
# Set environment
export ENV_FOR_DYNACONF=website
export GEMINI_API_KEY=your_key

# 1. Generate digest
labnotes --section ai_research_and_models --out ./out

# 2. Deduplicate similar articles
labnotes-dedup --input ./out --section ai_research_and_models

# 3. Summarize articles
labnotes-summarise --input ./out --section ai_research_and_models

# 4. Collect results
labnotes-collect --input ./out

# 5. Publish (optional)
labnotes-post
```

Or use the provided scripts:

```bash
./labnotes/scripts/exec_website.sh
./labnotes/scripts/exec_mountainmap.sh
```

## File Structure

```
labnotes/
├── digest.py          # Main digest generation
├── deduplicate.py     # Similarity-based deduplication
├── summarise.py       # AI summarization
├── collect.py         # Result collection
├── post.py            # Publishing to Slack/LinkedIn
├── embeddings.py      # Embedding generation (LiteLLM)
├── similarity.py      # Similarity detection
├── scraping.py        # Web content scraping
├── settings/
│   └── settings.yaml  # Centralized configuration
├── data/
│   └── feeds.yaml     # RSS/Atom feed configuration
└── scripts/
    ├── exec_website.sh
    └── exec_mountainmap.sh
```

## Development

```bash
# Install in development mode
uv sync

# Run with debug logging
labnotes --section ai_research_and_models --log-level DEBUG
```
