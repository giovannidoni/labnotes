import argparse
import json
import os
import re
from datetime import datetime


def sanitize_filename(filename):
    """Removes characters that aren't allowed in file names."""
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip()[:100].replace(" ", "_")


def process_conversations(input_path, output_dir, min_msgs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: {e}")
        return

    count = 0
    skipped = 0

    for chat in data:
        # --- NEW: Extract UUID ---
        chat_id = chat.get("id", "unknown_id")
        title = chat.get("title") or "Untitled Chat"
        mapping = chat.get("mapping", {})

        messages = []
        for node_id in mapping:
            node = mapping[node_id]
            if node.get("message") and node["message"].get("content"):
                content = node["message"]["content"]
                if content.get("content_type") == "text":
                    text = "".join(content.get("parts", []))
                    if text.strip():
                        role = node["message"]["author"]["role"]
                        time = node["message"]["create_time"]
                        messages.append({"role": role, "text": text, "time": time})

        if len(messages) < min_msgs:
            skipped += 1
            continue

        messages.sort(key=lambda x: x["time"] if x["time"] else 0)
        created_at = chat.get("create_time")
        date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d") if created_at else "Unknown"

        clean_title = sanitize_filename(title)
        filename = f"{date_str}_{clean_title}.md"

        # --- NEW: Add chat_id to Frontmatter ---
        md_content = f'---\ntitle: "{title}"\ncreated: {date_str}\nchat_id: "{chat_id}"\ntype: chatgpt-archive\n---\n\n'

        for msg in messages:
            label = "USER" if msg["role"] == "user" else "ASSISTANT"
            md_content += f"## {label}\n{msg['text']}\n\n"

        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as md_file:
            md_file.write(md_content)
        count += 1

    print(f"Exported: {count} chats with UUIDs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--min", type=int, default=6, help="Minimum number of messages to include a chat")
    args = parser.parse_args()
    process_conversations(args.input, args.output, args.min)
