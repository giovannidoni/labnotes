import json
import logging
import os
import traceback
from typing import Any, Dict, List

import aiofiles
from supabase import create_client

logger = logging.getLogger(__name__)


def save_output(items: List[Dict[str, Any]], output_path: str) -> None:
    """Save deduplicated items to JSON file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(items)} items to {output_path}")
    except Exception:
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
    except Exception:
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
    except Exception:
        error = traceback.format_exc()
        logger.error(f"Failed to write file {filepath}: {error}")
        raise
