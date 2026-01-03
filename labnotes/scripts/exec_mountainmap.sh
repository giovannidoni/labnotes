sections=(
    "mountainmap"
)

for section in "${sections[@]}"
do
    echo "Running labnotes for section: $section"
    uv run labnotes --section $section --scraper beautifulsoup --out ./out --log-level INFO
done

for section in "${sections[@]}"
do
    echo "Running labnotes summarise for section: $section"
    uv run labnotes-summarise --input ./out --section $section --log-level INFO
done
