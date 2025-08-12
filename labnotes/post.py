import requests
import traceback
import logging
import sys
import os
from labnotes.utils import load_input, setup_logging

logger = logging.getLogger(__name__)

head = """"Here's your daily AI digest 🤖, available now also at giovannidoni.github.io

Headlines 💡:"""


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


def get_article_block_text(item, i):
    """Generate a Slack block for an article."""
    return """
{i}) {summary}: {link}
    """.format(i=i, summary=item['summary'], link=item['link'])


def get_digest_block_text(digest):
    """Generate a Slack block for digest."""
    return """Digest ⚙️🧠:\n
{digest}
    """.format(digest=digest.replace("*", ""))


def get_digest_block(digest):
    """Generate a Slack block for digest."""
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*📋 Digest:*\n{digest}"
        }
    }


def get_slack_blocks(data):
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

    return blocks


def get_linkedin_block(data):
    """Generate a LinkedIn block for an article."""
    blocks = head + "\n"
    for i, item in enumerate(data["picked_headlines"]):
        blocks += get_article_block_text(item, i + 1)

    blocks += "\n"
    blocks += get_digest_block_text(data["digest"])

    return blocks


def post_to_slack(blocks):
    """Post the generated blocks to Slack."""
    res = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={
            "Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}", 
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


def post_to_linkedin(blocks):
    """Post the generated blocks to LinkedIn."""
    print(os.getenv('LINKEDIN_API_TOKEN'))
    headers = {
        "Authorization": f"Bearer {os.getenv('LINKEDIN_API_TOKEN')}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"        
    }
    PERSON_URN = f"urn:li:person:{os.getenv('PERSON_URN')}"
    payload = {
        "author": PERSON_URN,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": blocks
                },
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }
    
    res = requests.post("https://api.linkedin.com/v2/ugcPosts", headers=headers, json=payload)

    if res.status_code == 200:
        response_data = res.json()
        if response_data.get("ok"):
            logger.info("Message sent successfully to LinkedIn")
        else:
            logger.error(f"Slack API error: {response_data.get('error')}")
    else:
        logger.error(f"HTTP error: {res.status_code} - {res.text}")


def _main():
    """Main function to generate Slack message blocks."""
    data = load_input("./out/summarised_results.json")
    
    # Build blocks array properly
    blocks = get_slack_blocks(data)

    # Send to Slack
    # post_to_slack(blocks)

    # Build block for LinkedIn post
    text = get_linkedin_block(data)

    # Send to LinkedIn
    post_to_linkedin(blocks)

    logger.info(f"LinkedIn post content generated: {text}")


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
