from typing import List, Dict, Any
import logging
from pathlib import Path
import json
import yaml
import os
import traceback


logger = logging.getLogger(__name__)


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


def get_feed_sections(pipeline: str = "website") -> List[str]:
    """Get the list of feed sections from feeds.yaml."""
    feeds_file = Path(__file__).parent / "settings" / pipeline / "feeds.yaml"
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


def find_most_recent_file(directory: Path, pattern: str = "*.json") -> Path:
    """Find the most recent file matching the pattern in the given directory."""
    files = list(directory.glob(pattern))
    if not files:
        logger.warning(f"No files matching '{pattern}' found in directory: {directory}")
        return None

    # Return the file with the most recent modification time
    return max(files, key=lambda f: f.stat().st_mtime)
