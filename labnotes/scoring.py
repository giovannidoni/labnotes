from typing import Dict, Any
import logging


logger = logging.getLogger(__name__)


def score_item_for_audience(item: Dict[str, Any], audience_kw: Dict[str, Any], audience_name: str) -> Dict[str, Any]:
    """Score an item for a specific audience and return detailed scoring info."""
    text = f"{item['title']} {item.get('summary', '')} {item.get('content', '')}".lower()
    score = 0

    must_matches = []
    for k in audience_kw.get("must", []):
        if k and k.lower() in text:
            score += 2
            must_matches.append(k)

    nice_matches = []
    for k in audience_kw.get("nice", []):
        if k and k.lower() in text:
            score += 1
            nice_matches.append(k)

    src = item.get("source", "")
    source_adjustments = []
    if any(d in src for d in audience_kw.get("source_weight", {}).get("plus", [])):
        score += 1
        source_adjustments.append("+1 (source bonus)")
    if any(d in src for d in audience_kw.get("source_weight", {}).get("minus", [])):
        score -= 1
        source_adjustments.append("-1 (source penalty)")

    logger.debug(
        f"Scoring item '{item['title'][:50]}...' for {audience_name}: score={score}, "
        f"must_matches={must_matches}, nice_matches={nice_matches}, "
        f"source_adjustments={source_adjustments}"
    )

    return {
        "score": score,
        "must_matches": must_matches,
        "nice_matches": nice_matches,
        "source_adjustments": source_adjustments,
    }


def score_item_dual(item: Dict[str, Any], audience_kw: Dict[str, Any]) -> Dict[str, Any]:
    """Score an item for both engineers and managers, returning detailed scores."""
    scores = {}
    # New multi-audience format
    for audience_name, audience_kw in audience_kw.items():
        scores[audience_name] = score_item_for_audience(item, audience_kw, audience_name)

    return scores


def scoring(items: list[Dict[str, Any]], audience_kw: Dict[str, Any], top: int):
    """Score items based on keywords for different audiences."""
    logger.info("Starting dual-audience item scoring...")
    for it in items:
        scores = score_item_dual(it, audience_kw)
        it["_score"] = 0
        for k in audience_kw:
            it[f"_score_{k}"] = scores.get(k, {}).get("score", 0)
            it["_score"] += it[f"_score_{k}"]

    # Sort by combined score (engineers + managers)
    items.sort(key=lambda x: x["_score"], reverse=True)
    top = items[:top]

    # Log scores for both audiences
    for k in audience_kw:
        this_scores = [item[f"_score_{k}"] for item in top]
        logger.info(f"{k.capitalize()} scores: {this_scores}")
    return top
