sections=(
    "mountainmap"
)
TOP=${1:-100}
echo $TOP
for section in "${sections[@]}"
do
    echo "Running labnotes for section: $section"
    uv run labnotes --hours 168 --top $TOP --section $section --scraper beautifulsoup --out ./out --log-level INFO
done

for section in "${sections[@]}"
do
    echo "Running labnotes summarise for section: $section"
    uv run labnotes-summarise --input ./out --section $section --model gpt-4o --log-level INFO
done
