from typing import List, Dict, Any
import logging
from pathlib import Path
import json
import yaml
import os
import traceback
import aiofiles

from supabase import create_client

logger = logging.getLogger(__name__)


def save_output(items: List[Dict[str, Any]], output_path: str) -> None:
    """Save deduplicated items to JSON file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(items)} items to {output_path}")
    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Failed to save file {output_path}: {error}")
        raise


def load_input(filepath: str) -> List[Dict[str, Any]]:
    """Load JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            items = json.load(f)

        logger.info(f"Loaded {len(items)} items from {filepath}")
        return items
    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Failed to load JSON file {filepath}: {error}")
        raise

def save_to_supabase(items: Dict, table_name: str) -> None:
    """Save items to Supabase database."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_TOKEN")
    supabase = create_client(url, key)
    rows = [{"rec": obj} for obj in items]

    # Insert (or upsert if you created unique index on link+group)
    try:
        _ = supabase.table(table_name).insert(rows).execute()
        logger.info(f"Inserted {len(items)} items into Supabase")
    except Exception as _:
        error = traceback.format_exc()
        logger.info(f"Failed to insert items into Supabase: {error}")


async def write_file(filepath: str, content: str) -> None:
    """Write content to file asynchronously."""
    try:
        async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
            await f.write(content)
        logger.debug(f"Successfully wrote file: {filepath}")
    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Failed to write file {filepath}: {error}")
        raise


def find_most_recent_file(directory: Path, pattern: str = "*.json") -> Path:
    """Find the most recent file matching the pattern in the given directory."""
    files = list(directory.glob(pattern))
    if not files:
        logger.warning(f"No files matching '{pattern}' found in directory: {directory}")
        return None

    # Return the file with the most recent modification time
    return max(files, key=lambda f: f.stat().st_mtime)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("aiohttp").setLevel(getattr(logging, level.upper(), None))
    logging.getLogger("urllib3").setLevel(getattr(logging, level.upper(), None))
    logging.getLogger("requests").setLevel(getattr(logging, level.upper(), None))
    logging.getLogger("sentence_transformers").setLevel(getattr(logging, level.upper(), None))
    logging.getLogger("transformers").setLevel(getattr(logging, level.upper(), None))
    logging.getLogger("torch").setLevel(getattr(logging, level.upper(), None))


def get_feed_sections() -> List[str]:
    """Get the list of feed sections from feeds.yaml."""
    feeds_file = Path(__file__).parent / "data" / "feeds.yaml"
    try:
        with open(feeds_file, "r", encoding="utf-8") as f:
            feeds_data = yaml.safe_load(f)

        if not feeds_data:
            logger.warning("Feeds file is empty")
            return []

        sections = list(feeds_data.keys())
        logger.debug(f"Found {len(sections)} feed sections: {sections}")
        return sections

    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Failed to load feed sections: {error}")
        return []
