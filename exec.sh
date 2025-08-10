sections=(
    "ai_research_and_models"
    "community_signals"
    "engineering_leadership_and_management"
)
# TOP=${1:-100}
# for section in "${sections[@]}"
# do
#     echo "Running labnotes for section: $section"
#     uv run labnotes --hours 24 --top $TOP --section $section --out ./out --log-level INFO
# done

# for section in "${sections[@]}"
# do
#     echo "Running labnotes dedub for section: $section"
#     uv run labnotes-dedup --input ./out --section $section --threshold 0.7 --prefer-score --log-level INFO
# done

for section in "${sections[@]}"
do
    echo "Running labnotes for section: $section"
    uv run labnotes-summarise --input ./out --section $section --log-level INFO
done

 uv run labnotes-collect --input ./out --log-level INFO