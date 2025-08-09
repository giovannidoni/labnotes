# AI Daily Digest (uv-native)

Python-only daily AI digest. Pulls curated RSS/Atom feeds, filters last 24h, dedupes, scores for relevance, scrapes full article content, and writes a ranked digest as **Markdown and JSON**.

## Quick start (with `uv`)
```bash
# Install deps into a virtual env
uv sync

# Run with defaults (24h window, top 3 items, outputs to ./out)
uv run python app/digest.py

# Customise
uv run python app/digest.py --hours 24 --top 5 --format md,json,txt --out ./out

# Process only one section of feeds
uv run python app/digest.py --section "AI Research & Models"
```

## Features
- **RSS/Atom parsing**: Fetches from curated feeds
- **Web scraping**: Extracts full article content from each link
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
