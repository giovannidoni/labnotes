# ChatGPT Knowledge Base Pipeline

Two-step pipeline to transform ChatGPT exports into searchable, AI-curated Obsidian notes.

## Overview

```
conversations.json → extract_chats.py → Raw Markdown → curate_knowledge_base.py → Curated KB Notes
```

## Scripts

### 1. `extract_chats.py` - Export Conversations

Converts ChatGPT JSON export into individual markdown files.

**Usage:**
```bash
python extract_chats.py <input.json> <output_dir> [--min N]
```

**Arguments:**
- `input`: ChatGPT export JSON file (download from ChatGPT settings)
- `output`: Destination folder for markdown files
- `--min`: Minimum messages per chat (default: 4)

**Output:** Markdown files with frontmatter containing `chat_id`, `title`, and `created` date.

**Example:**
```bash
python extract_chats.py ~/Downloads/conversations.json ~/rawKnowledgeBase/ --min 3
```

---

### 2. `curate_knowledge_base.py` - AI Curation

Uses Gemini 3 Flash to intelligently curate raw conversations into structured notes.

**Usage:**
```bash
python curate_knowledge_base.py <source> <dest> [--limit N] [--overwrite]
```

**Arguments:**
- `source`: Folder with raw markdown exports
- `dest`: Obsidian vault knowledge base folder
- `--limit N`: Process only first N files (optional)
- `--overwrite`: Force reprocessing, delete existing notes with same chat_id

**Features:**
- Filters trivial/sensitive/redundant conversations
- Extracts summary, insights, outcomes, solutions
- Splits multi-topic conversations into separate notes
- Formats with Obsidian callouts and kebab-case tags
- Skips duplicates automatically (unless `--overwrite`)
- Organizes by date (`YYYY/MM/`)

**Output Structure:**
```markdown
---
title: "descriptiveTitle"
date: 2025-01-10
tags: [python-async, performance]
chat_id: "uuid"
---

## Summary
Brief overview of accomplishments.

> [!todo] **Session Intel**
> **Goal:** Root intent
> **Insights:** Key findings
> **Outcomes:** ✅ Results
> **💡 Solution:** Final answer

> [!quote]- Full Transcript
> **User**: Question
> **AI**: Response
```

**Example:**
```bash
# Test with first 5 files
python curate_knowledge_base.py ~/rawKnowledgeBase/ ~/Obsidian/KB --limit 5

# Process all new chats
python curate_knowledge_base.py ~/rawKnowledgeBase/ ~/Obsidian/KB

# Reprocess and overwrite existing
python curate_knowledge_base.py ~/rawKnowledgeBase/ ~/Obsidian/KB --overwrite
```

## Complete Workflow

1. **Export from ChatGPT:** Download `conversations.json` from ChatGPT settings
2. **Extract:** `python extract_chats.py conversations.json ~/rawKnowledgeBase/`
3. **Curate:** `python curate_knowledge_base.py ~/rawKnowledgeBase/ ~/Obsidian/KB --limit 5` (test)
4. **Full run:** `python curate_knowledge_base.py ~/rawKnowledgeBase/ ~/Obsidian/KB`

## Installation

```bash
uv pip install litellm tqdm
```

Configure Gemini API credentials for litellm.

## Notes

- **Auto-filtering:** Removes trivial, iterative, ephemeral, or sensitive chats
- **Email redaction:** Automatically redacts email addresses
- **Cost tracking:** Shows token usage and estimated costs
- **Duplicate handling:** `--overwrite` deletes ALL files with matching chat_id before recreating
- **Rate limiting:** 0.5s delay between API calls
