import requests
import traceback
import logging
import sys
import os
from labnotes.utils import load_input, setup_logging

logger = logging.getLogger(__name__)


def get_article_block(item):
    """Generate a Slack block for an article."""
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*{item['summary']}*"
        },
        "accessory": {
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": "Read More"
            },
            "url": item['link']
        }
    }


def get_digest_block(digest):
    """Generate a Slack block for digest."""
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*📋 Digest:*\n{digest}"
        }
    }


def _main():
    """Main function to generate Slack message blocks."""
    data = load_input("./out/summarised_results.json")
    
    # Build blocks array properly
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Labnotes 📰",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Here's your daily AI research digest :robot_face:"
            }
        },
        {"type": "divider"}
    ]
    
    # Add article blocks
    for item in data["picked_headlines"]:
        blocks.append(get_article_block(item))
    
    # Add digest
    blocks.append({"type": "divider"})
    blocks.append(get_digest_block(data["digest"]))

    # Send to Slack
    res = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={
            "Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}", 
            "Content-Type": "application/json; charset=utf-8"
        },
        json={
            "channel": "C09A405DMPB", 
            "blocks": blocks  # Pass blocks array directly
        }
    )
    
    if res.status_code == 200:
        response_data = res.json()
        if response_data.get("ok"):
            logger.info("Message sent successfully to Slack")
        else:
            logger.error(f"Slack API error: {response_data.get('error')}")
    else:
        logger.error(f"HTTP error: {res.status_code} - {res.text}")


def main():
    """Synchronous CLI entry point wrapper."""
    try:
        setup_logging()
        _main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Unexpected error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
