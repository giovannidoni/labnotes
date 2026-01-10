import os, json, re, argparse, time
from litellm import completion 
from datetime import datetime

# --- CONFIG ---
# GEMINI_API_KEY must be in your environment variables.

def extract_date_from_filename(filename):
    """Parses YYYY-MM-DD from the source filename."""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    return match.group(1) if match else datetime.now().strftime('%Y-%m-%d')

def extract_uuid_from_content(content):
    """Parses the chat_id from the Stage 1 Markdown frontmatter."""
    match = re.search(r'chat_id:\s*"(.*?)"', content)
    return match.group(1) if match else "unknown"

def local_redactor(text):
    """Masks emails locally before sending to API."""
    email_re = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.sub(email_re, "[EMAIL_REDACTED]", text)

def to_camel_case(text):
    """Converts tags to Obsidian-friendly camelCase."""
    words = re.findall(r'[a-zA-Z0-9]+', text)
    return words[0].lower() + "".join(w.capitalize() for w in words[1:]) if words else ""

def curate_with_gemini_3(content):
    """Gemini 3 Flash: Curation logic with structured insight extraction."""
    prompt = f"""
    # ROLE
    Digital Archivist & Knowledge Engineer.
    
    # INSTRUCTIONS
    1. FILTER: Return {{"action": "DELETE"}} if the chat meet one of the following:
       - TRIVIAL: Lacks substantive content (e.g., too general, too short and query on a file).
       - ITERATIVE NOISE: Contains coding iterations that are too specific/granular.
       - EPHEMERAL/NICHE: Primarily about a niche topic or mundane logistics.
       - SIMPLE UTILITY: Simple translation, proofreading, or copy-editing.
    2. DISTILL:
       - inquiry: 1-sentence summary of intent.
       - findings: 2-3 key insights discovered.
       - outcomes: 2-3 actionable results.
       - solution: The final, corrected answer.
    3. TRANSCRIPT: Use nested callouts for User/AI dialogue.
    
    # OUTPUT FORMAT
    Return ONLY JSON:
    [{{
      "title": "...", # Short title, use camelCase or underscores
      "tags": ["..."],
      "inquiry": "...",
      "findings": ["...", "..."],
      "outcomes": ["...", "..."],
      "solution": "...",
      "transcript": "...",
      "action": "KEEP"
    }}]

    # CONTENT
    {content}
    """
    try:
        response = completion(
            model="gemini/gemini-3-flash-preview", 
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            reasoning_effort="low",
            temperature=1.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None

def main(source, dest):
    os.makedirs(dest, exist_ok=True)
    files = [f for f in os.listdir(source) if f.endswith(".md")]
    
    print(f"🚀 Processing {len(files)} files for the vault...")

    for filename in files:
        original_date = extract_date_from_filename(filename)
        # Parse year and month for subfolders
        date_obj = datetime.strptime(original_date, '%Y-%m-%d')
        year_str = date_obj.strftime('%Y')
        month_str = date_obj.strftime('%m')
        
        # Create target subfolder: dest/YYYY/MM
        target_dir = os.path.join(dest, year_str, month_str)
        os.makedirs(target_dir, exist_ok=True)

        path = os.path.join(source, filename)
        with open(path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        current_uuid = extract_uuid_from_content(raw_content)
        redacted_text = local_redactor(raw_content)

        print(f"🧠 Distilling into {year_str}/{month_str}: {filename}")
        results = curate_with_gemini_3(redacted_text)

        if results and isinstance(results, list):
            for i, item in enumerate(results):
                if item.get("action") == "DELETE": continue
                
                safe_title = re.sub(r'[\\/*?:"<>|]', "", item['title'])[:50].strip().replace(" ", "_")
                suffix = f"_part{i+1}" if len(results) > 1 else ""
                output_fn = f"{original_date}_{safe_title}{suffix}.md"
                output_path = os.path.join(target_dir, output_fn)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("---\n")
                    f.write(f"title: \"{item['title']}\"\n")
                    f.write(f"date: {original_date}\n")
                    f.write(f"tags: {[to_camel_case(t) for t in item.get('tags', [])]}\n")
                    f.write(f"chat_id: \"{current_uuid}\"\n")
                    f.write(f"original_source: \"{filename}\"\n")
                    f.write("---\n\n")

                    f.write(f"> [!todo] **Session Intel**\n")
                    f.write(f"> **Goal:** {item.get('inquiry', '')}\n")
                    f.write(f"> \n")
                    f.write(f"> **Insights:**\n")
                    for fnd in item.get('findings', []):
                        f.write(f"> - {fnd}\n")
                    f.write(f"> \n")
                    f.write(f"> **Outcomes:**\n")
                    for out in item.get('outcomes', []):
                        f.write(f"> - ✅ {out}\n")
                    f.write(f"> \n")
                    f.write(f"> **💡 Final Solution:**\n")
                    indented_sol = item.get('solution', '').replace("\n", "\n> ")
                    f.write(f"> {indented_sol}\n\n")

                    f.write(f"> [!quote]- Full Transcript (Source Context)\n")
                    nested_chat = item.get('transcript', '').replace("\n", "\n> ")
                    f.write(f"> {nested_chat}\n")

                ts = date_obj.timestamp()
                os.utime(output_path, (ts, ts))
        
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path to staging folder")
    parser.add_argument("dest", help="Path to obsidian vault")
    args = parser.parse_args()
    main(args.source, args.dest)