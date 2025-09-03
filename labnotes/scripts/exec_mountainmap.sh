sections=(
    "mountainmap"
)
TOP=${1:-100}
echo $TOP
for section in "${sections[@]}"
do
    echo "Running labnotes for section: $section"
    uv run labnotes --hours 24 --top $TOP --section $section --out ./out --log-level INFO
done

for section in "${sections[@]}"
do
    echo "Running labnotes summarise for section: $section"
    uv run labnotes-summarise --input ./out --section $section --log-level INFO
done
