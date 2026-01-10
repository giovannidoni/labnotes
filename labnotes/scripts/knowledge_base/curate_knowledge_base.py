import argparse
import json
import logging
import os
import re
import time
from datetime import datetime

import litellm
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# --- LOGGING SETUP ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)


def extract_date_from_filename(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else datetime.now().strftime("%Y-%m-%d")


def extract_uuid_from_content(content):
    match = re.search(r'chat_id:\s*"(.*?)"', content)
    return match.group(1) if match else "unknown"


def local_redactor(text):
    email_re = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return re.sub(email_re, "[EMAIL_REDACTED]", text)


def to_camel_case(text):
    words = re.findall(r"[a-zA-Z0-9]+", text)
    return words[0].lower() + "".join(w.capitalize() for w in words[1:]) if words else ""


def to_kebab_case(text):
    words = re.findall(r"[a-zA-Z0-9]+", text)
    return "-".join(w.lower() for w in words) if words else ""


def chat_id_exists(dest, chat_id):
    """Check if any file in dest contains the given chat_id. Returns list of filepaths if found, empty list otherwise."""
    matching_files = []
    for root, dirs, files in os.walk(dest):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read(500)  # Only read frontmatter
                        if f'chat_id: "{chat_id}"' in content:
                            matching_files.append(filepath)
                except:
                    continue
    return matching_files


def curate_with_gemini_3(content):
    """Gemini 3 Flash: Curation logic with structured insight extraction."""
    prompt = f"""
    # ROLE
    Digital Archivist & Knowledge Engineer.
    
    # INSTRUCTIONS
    1. **FILTER**: Return {{"action": "DELETE"}} if the chat meets any:
       - TRIVIAL: Lacks substantive content.
       - ITERATIVE NOISE: Granular coding iterations or redundant debugging.
       - EPHEMERAL/NICHE: Mundane logistics or zero future value.
       - SENSITIVE DATA: Contains keys, secrets, or PII.
       - SIMPLE UTILITY: Simple translation or copy-editing.
       - CODING QUESTIONS: Chat requests for code edits or to generate code should be skipped

    2. **MULTI-TOPIC SPLITTING**: 
       - If the user switches between unrelated technical topics, return a SEPARATE JSON object for each in the array.
       - Make sure not to fragment coherent sessions, and not to generate too many splits; short chats should remain single objects, are were better removed if not valuable.
       - If the chat is one continuous flow, return a single-object array.

    3. **DISTILL (Per Object)**:
       - summary: A concise 2-3 sentence overview of what was accomplished in this session.
       - inquiry: 1-sentence summary of the specific root intent.
       - findings: 2-3 key technical insights.
       - outcomes: 2-3 actionable results/fixes achieved.
       - solution: The final, optimized definitive answer or code.

    4. **TRANSCRIPT (Per Object)**: 
       - Relevant dialogue ONLY. Format: `**User**: ...` and `**AI**: ...`.
       - Leave a blank line between exchanges.
       - If a chat was interrupted, skip the incomplete turn.
    
    # OUTPUT FORMAT
    Return ONLY JSON:
    [{{
      "title": "...", # Short title, use camelCase, no special chars, max 60 chars
      "tags": ["..."], # each tag must be "-" separated, no spaces
      "summary": "...",      
      "inquiry": "...",
      "findings": ["...", "..."],
      "outcomes": ["...", "..."],
      "solution": "...",
      "transcript": "...",
      "action": "KEEP" | "DELETE"
    }}]

    # CONTENT
    {content}
    """
    try:
        response = litellm.completion(
            model="gemini/gemini-3-flash-preview",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=1.0,
        )

        # Token usage tracking
        usage = response.usage
        token_stats = {
            "prompt": usage.prompt_tokens,
            "completion": usage.completion_tokens,
            "total": usage.total_tokens,
        }

        res_data = json.loads(response.choices[0].message.content)
        # Ensure array format
        results = res_data if isinstance(res_data, list) else res_data.get("data", [res_data])

        return results, token_stats
    except Exception as e:
        logger.error(f"❌ API Error: {e}")
        return None, None


def main(source, dest, limit=None, overwrite=False):
    os.makedirs(dest, exist_ok=True)
    files = [f for f in os.listdir(source) if f.endswith(".md")]

    # Tracking aggregates
    totals = {"prompt": 0, "completion": 0, "files": 0, "skipped": 0}

    total_files = len(files) if limit is None else min(limit, len(files))
    logging.info(f"🚀 Processing {total_files} files...")

    # Redirect logging so it doesn't break tqdm bar
    with logging_redirect_tqdm():
        files_to_process = files[:limit] if limit else files
        for filename in tqdm(files_to_process, desc="📂 Archiving", unit="chat"):
            original_date = extract_date_from_filename(filename)
            date_obj = datetime.strptime(original_date, "%Y-%m-%d")
            target_dir = os.path.join(dest, date_obj.strftime("%Y"), date_obj.strftime("%m"))

            with open(os.path.join(source, filename), "r", encoding="utf-8") as f:
                raw_content = f.read()

            os.makedirs(target_dir, exist_ok=True)
            current_uuid = extract_uuid_from_content(raw_content)

            # Check if file with this chat_id already exists
            existing_files = chat_id_exists(dest, current_uuid)

            if existing_files:
                if overwrite:
                    # Delete all existing files with this chat_id before processing
                    for existing_file in existing_files:
                        os.remove(existing_file)
                        logger.debug(f"Removed existing file: {existing_file}")
                else:
                    # Skip processing
                    totals["skipped"] += 1
                    continue

            results, tokens = curate_with_gemini_3(local_redactor(raw_content))

            if results and tokens:
                totals["prompt"] += tokens["prompt"]
                totals["completion"] += tokens["completion"]
                totals["files"] += 1

                for i, item in enumerate(results):
                    if item.get("action") == "DELETE":
                        continue

                    safe_title = re.sub(r'[\\/*?:"<>|]', "", item["title"]).strip().replace(" ", "_")
                    suffix = f"_part{i + 1}" if len(results) > 1 else ""
                    output_fn = f"{original_date}_{safe_title}{suffix}.md"
                    output_path = os.path.join(target_dir, output_fn)

                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write("---\n")
                        f.write(f'title: "{item["title"]}"\n')
                        f.write(f"date: {original_date}\n")
                        f.write(f"tags: {[to_kebab_case(t) for t in item.get('tags', [])]}\n")
                        f.write(f'chat_id: "{current_uuid}"\n')
                        f.write(f'original_source: "{filename}"\n')
                        f.write("---\n\n")

                        f.write("## Summary\n\n")
                        f.write(f"{item.get('summary', '')}\n\n")

                        f.write("> [!todo] **Session Intel**\n")
                        f.write(f"> **Goal:** {item.get('inquiry', '')}\n>\n")
                        f.write("> **Insights:**\n")
                        for fnd in item.get("findings", []):
                            f.write(f"> - {fnd}\n")
                        f.write("> \n> **Outcomes:**\n")
                        for out in item.get("outcomes", []):
                            f.write(f"> - ✅ {out}\n")

                        sol = item.get("solution", "")
                        if sol:
                            sol = "> " + sol.replace("\n", "\n> ")
                        f.write(f"> \n> **💡 Solution:**\n{sol}\n\n")

                        trans = item.get("transcript", "")
                        if trans:
                            trans = "> " + trans.replace("\n", "\n> ")
                        f.write(f"> [!quote]- Full Transcript\n{trans}\n")

                    os.utime(output_path, (date_obj.timestamp(), date_obj.timestamp()))

                time.sleep(0.5)

    # --- FINAL REPORT ---
    print("\n" + "=" * 40)
    print(f"🏁 DONE: {totals['files']} files processed, {totals['skipped']} skipped.")
    print(f"📊 Prompt Tokens: {totals['prompt']} | Completion: {totals['completion']}")
    print(f"💰 Est. Cost: ${(totals['prompt'] * 0.075 + totals['completion'] * 0.3) / 1_000_000:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path to staging folder")
    parser.add_argument("dest", help="Path to obsidian vault")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files (default: process all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files (default: skip existing)")
    args = parser.parse_args()
    main(args.source, args.dest, args.limit, args.overwrite)
