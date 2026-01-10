uv run labnotes-aria-padana  

src=./out
dst=~/Google\ Drive/Personal\&Projects/aria.padana/
today=$(date +%Y-%m-%d)

mkdir -p "$dst"
for f in "$src"/*; do
  [ -f "$f" ] && cp "$f" "$dst/${today}_$(basename "$f")"
done
