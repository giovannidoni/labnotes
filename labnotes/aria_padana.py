"""
Aria Padana - Air Quality monitoring for Northern Italy (Po Valley).

This module fetches air quality data and images from the European Environment Agency's
ArcGIS service and posts updates to Instagram.
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime as dt
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont

from labnotes.tools.utils import setup_logging

logger = logging.getLogger(__name__)

# Bounding boxes for cities in Northern Italy (EPSG:3857)
CITY_BBOXES = {
    "Torino": "845702,5622596,865702,5642596",
    "Milano": "1013026,5684899,1033026,5704899",
    "Genova": "985898,5518426,1005898,5538426",
    "Brescia": "1126772,5697192,1146772,5717192",
    "Bergamo": "1067272,5722133,1087272,5742133",
    "Verona": "1213657,5680805,1233657,5700805",
    "Venezia": "1360955,5681185,1380955,5701185",
    "Bologna": "1252652,5532351,1272652,5552351",
    "Parma": "1139697,5580326,1159697,5600326",
    "Trieste": "1523626,5714358,1543626,5734358",
    "Trento": "1227995,5782344,1247995,5802344",
    "Udine":  "1463267,5781745,1483267,5801745",
    "Bolzano": "1254006,5850564,1274006,5870564",
    "Ferrara": "1283509,5586073,1303509,5606073"
}


# City center coordinates (EPSG:3857) - center of bounding boxes
CITY_CENTERS = {
    "Torino": (855702, 5632596),
    "Milano": (1023026, 5694899),
    "Genova": (995898, 5528426),
    "Brescia": (1136772, 5707192),
    "Bergamo": (1077272, 5732133),
    "Verona": (1223657, 5690805),
    "Venezia": (1370955, 5691185),
    "Bologna": (1262652, 5542351),
    "Parma": (1149697, 5590326),
    "Trieste": (1533626, 5724358),
    "Trento": (1237995, 5792344),
    "Udine":   (1473267, 5791745),
    "Bolzano": (1264006, 5860564),
    "Ferrara": (1293509, 5596073)
}


# AQI level descriptions (European AQI scale) - in Italian
AQI_LEVELS = {
    1: ("Buona", "🟢"),
    2: ("Discreta", "🟡"),
    3: ("Moderata", "🟠"),
    4: ("Scarsa", "🔴"),
    5: ("Molto Scarsa", "🟣"),
    6: ("Pessima", "⚫"),
}

# Base URLs for the EEA ArcGIS service
BASE_URL = "https://air.discomap.eea.europa.eu/arcgis/rest/services/AQMobile_2025/MOSAIC_GLOBAL_AQI/ImageServer"

# AQI colors (RGB) matching the EEA colormap
AQI_COLORS = {
    1: (80, 240, 230),   # Buona - cyan
    2: (80, 204, 170),   # Discreta - green
    3: (240, 230, 65),   # Moderata - yellow
    4: (255, 80, 80),    # Scarsa - red
    5: (150, 0, 50),     # Molto Scarsa - dark red
    6: (125, 33, 129),   # Pessima - purple
}

def add_legend_to_image(image_path: str, city_stats: list[dict] | None = None) -> None:
    """
    Add a semi-transparent legend overlay and city labels to the air quality image.

    Args:
        image_path: Path to the image file to modify.
        city_stats: List of city statistics from fetch_all_city_stats().
    """
    try:
        img = Image.open(image_path).convert("RGBA")

        # Current bounding box (must match fetch_air_quality_image)
        center_x = 1185000
        center_y = 5700000
        width = 750000
        height = width * 5 / 4
        bbox_x_min = center_x - width / 2
        bbox_x_max = center_x + width / 2
        bbox_y_min = center_y - height / 2
        bbox_y_max = center_y + height / 2

        # Legend dimensions - two rows, aligned columns
        legend_height = 120
        box_size = 36
        box_spacing = 10
        font_size = 28
        row_spacing = 14
        col_width = 340  # Fixed column width for alignment

        # Create overlay for legend background
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Semi-transparent dark background at bottom
        legend_y = img.height - legend_height
        draw.rectangle(
            [0, legend_y, img.width, img.height],
            fill=(0, 0, 0, 200)  # Semi-transparent black
        )

        # Try to use a nice font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        # Split items into two rows (3 items each)
        items = list(AQI_LEVELS.items())
        row1 = items[:3]
        row2 = items[3:]

        for row_idx, row_items in enumerate([row1, row2]):
            # Start position for row (centered)
            total_width = col_width * len(row_items)
            x_start = (img.width - total_width) // 2

            y_pos = legend_y + 14 + row_idx * (box_size + row_spacing)

            for i, (level, (label, _)) in enumerate(row_items):
                color = AQI_COLORS[level]
                col_x = x_start + i * col_width

                # Draw colored box
                draw.rectangle(
                    [col_x, y_pos, col_x + box_size, y_pos + box_size],
                    fill=color,
                    outline=(255, 255, 255, 220),
                    width=2
                )

                # Draw label in white
                label_y = y_pos + (box_size - font_size) // 2
                draw.text((col_x + box_size + box_spacing, label_y), label, fill=(255, 255, 255, 255), font=font)

        # Add title bar at top (two lines: title centered, date below)
        title_text = "Qualità dell'aria - @aria.padana"
        date_text = dt.now().strftime("%d/%m/%Y - %H:%M")
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
            date_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except (OSError, IOError):
            try:
                title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
                date_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
            except (OSError, IOError):
                title_font = font
                date_font = font

        title_bar_height = 70
        from_top = 8
        draw.rectangle([0, from_top, img.width, title_bar_height + from_top], fill=(0, 0, 0, 220))

        # Draw title centered
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_x = (img.width - (title_bbox[2] - title_bbox[0])) // 2
        draw.text((title_x, 14), title_text, fill=(255, 255, 255, 255), font=title_font)

        # Draw date/time centered below title (smaller font)
        date_bbox = draw.textbbox((0, 0), date_text, font=date_font)
        date_x = (img.width - (date_bbox[2] - date_bbox[0])) // 2
        draw.text((date_x, 48), date_text, fill=(180, 180, 180, 255), font=date_font)

        # Add stats panel below title (separate panel, similar to legend style)
        if city_stats:
            try:
                panel_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except (OSError, IOError):
                try:
                    panel_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
                except (OSError, IOError):
                    panel_font = font

            # Sort stats by AQI (worst first)
            sorted_stats = sorted(
                city_stats,
                key=lambda x: x.get("mean", 0) if x.get("mean") is not None else 0,
                reverse=True,
            )

            # Panel dimensions - similar to legend style
            box_size = 28
            line_height = 38
            num_cities = min(len(sorted_stats), 12)
            rows_per_col = (num_cities + 1) // 2
            panel_padding = 18
            panel_height = panel_padding * 2 + rows_per_col * line_height
            panel_top = title_bar_height + 16  # Gap between title and stats panel

            # Draw stats panel background (separate from title)
            draw.rectangle(
                [0, panel_top, img.width, panel_top + panel_height],
                fill=(0, 0, 0, 200)
            )

            # Draw stats in two columns with colored boxes (like legend)
            col1_x = 30
            col2_x = img.width // 2 + 20
            start_y = panel_top + panel_padding

            for i, stat in enumerate(sorted_stats[:num_cities]):
                city = stat["city"]
                mean_aqi = stat.get("mean")
                if mean_aqi is not None:
                    aqi_level = min(6, max(1, round(mean_aqi)))
                    color = AQI_COLORS.get(aqi_level, (128, 128, 128))
                    level_name, _ = AQI_LEVELS.get(aqi_level, ("?", "?"))

                    # Determine column and position
                    if i < rows_per_col:
                        col_x = col1_x
                        row = i
                    else:
                        col_x = col2_x
                        row = i - rows_per_col

                    y_pos = start_y + row * line_height

                    # Draw colored box (like legend)
                    draw.rectangle(
                        [col_x, y_pos, col_x + box_size, y_pos + box_size],
                        fill=color,
                        outline=(255, 255, 255, 200),
                        width=2
                    )

                    # Draw city name, level and value (like IG caption)
                    text = f"{city}: {level_name}"
                    text_y = y_pos + (box_size - 24) // 2
                    draw.text((col_x + box_size + 12, text_y), text, fill=(255, 255, 255, 255), font=panel_font)

        # Add city labels with AQI values
        if city_stats:
            try:
                city_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            except (OSError, IOError):
                try:
                    city_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
                except (OSError, IOError):
                    city_font = font

            # Create stats lookup
            stats_lookup = {s["city"]: s.get("mean") for s in city_stats}

            for city, (geo_x, geo_y) in CITY_CENTERS.items():
                # Convert geo coordinates to pixel coordinates
                px = int((geo_x - bbox_x_min) / (bbox_x_max - bbox_x_min) * img.width)
                py = int((bbox_y_max - geo_y) / (bbox_y_max - bbox_y_min) * img.height)

                # Skip if outside image bounds
                if px < 0 or px > img.width or py < 0 or py > img.height - legend_height:
                    continue

                # Get AQI value
                aqi_mean = stats_lookup.get(city)
                if aqi_mean is not None:
                    aqi_level = min(6, max(1, round(aqi_mean)))
                    label = f"{city}"

                    # Get text size
                    text_bbox = draw.textbbox((0, 0), label, font=city_font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    # Center label on city position
                    label_x = px - text_width // 2
                    label_y = py - text_height // 2

                    # Draw background
                    pad = 4
                    draw.rectangle(
                        [label_x - pad, label_y - pad, label_x + text_width + pad, label_y + text_height + pad],
                        fill=(0, 0, 0, 160)
                    )

                    # Draw text
                    draw.text((label_x, label_y), label, fill=(255, 255, 255, 255), font=city_font)

        # Composite overlay onto image
        result = Image.alpha_composite(img, overlay)

        # Save the modified image
        result.save(image_path, "PNG")
        logger.info(f"Added legend to image: {image_path}")

    except Exception as e:
        logger.error(f"Failed to add legend to image: {e}")


def fetch_air_quality_image(output_path: str = "./out/aria_padana.png", size: int = 1080) -> str | None:
    """
    Fetch the air quality map image for Northern Italy.

    Args:
        output_path: Path to save the downloaded image.
        size: Image size (width and height for square format).

    Returns:
        Path to the saved image, or None if download failed.
    """
    # Bounding box for Po Valley region - 4:5 aspect ratio (portrait for Instagram)
    # Zoomed in on Northern Italy
    center_x = 1185000  # Center on Po Valley
    center_y = 5700000
    width = 750000  # Narrower view
    height = width * 5 / 4  # 4:5 aspect ratio

    x_min = center_x - width / 2
    x_max = center_x + width / 2
    y_min = center_y - height / 2
    y_max = center_y + height / 2

    bbox = f"{x_min},{y_min},{x_max},{y_max}"

    # Image size: 1080x1350 for 4:5 aspect ratio
    img_width = size
    img_height = int(size * 5 / 4)

    params = {
        "bbox": bbox,
        "bboxSR": "",
        "size": f"{img_width},{img_height}",
        "imageSR": "",
        "datumTransformation": "",
        "time": "",
        "format": "jpgpng",
        "pixelType": "U8",
        "noData": "",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "compression": "",
        "compressionQuality": "",
        "bandIds": "",
        "sliceId": "",
        "mosaicRule": '{"ascending":false,"mosaicMethod":"esriMosaicAttribute","mosaicOperation":"MT_FIRST","sortField":"StdTime","sortValue":"0"}',
        "renderingRule": '{"rasterFunction":"Colormap","rasterFunctionArguments":{"Colormap":[[1,80,240,230],[2,80,204,170],[3,240,230,65],[4,255,80,80],[5,150,0,50],[6,125,33,129]]}}',
        "adjustAspectRatio": "true",
        "validateExtent": "false",
        "lercVersion": "1",
        "compressionTolerance": "",
        "f": "image",
    }

    try:
        response = requests.get(f"{BASE_URL}/exportImage", params=params, timeout=30)
        response.raise_for_status()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Air quality image saved to {output_path}")
        return output_path

    except requests.RequestException as e:
        logger.error(f"Failed to fetch air quality image: {e}")
        return None


def fetch_city_stats(city: str, bbox: str) -> dict[str, Any] | None:
    """
    Fetch air quality statistics for a specific city.

    Args:
        city: Name of the city.
        bbox: Bounding box coordinates in format "xmin,ymin,xmax,ymax".

    Returns:
        Dictionary with statistics, or None if request failed.
    """
    params = {
        "geometry": bbox,
        "geometryType": "esriGeometryEnvelope",
        "sr": "3857",
        "pixelSize": "500",
        "f": "json",
    }

    try:
        response = requests.get(f"{BASE_URL}/computeStatisticsHistograms", params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Extract relevant statistics
        stats = {
            "city": city,
            "raw_data": data,
        }

        # Parse histogram data if available
        if "histograms" in data and data["histograms"]:
            histogram = data["histograms"][0]
            counts = histogram.get("counts", [])

            # Find dominant AQI level (index with highest count)
            if counts:
                dominant_idx = counts.index(max(counts))
                stats["dominant_aqi"] = dominant_idx + 1  # AQI levels are 1-indexed
                stats["histogram_counts"] = counts

        # Parse statistics if available
        if "statistics" in data and data["statistics"]:
            stat = data["statistics"][0]
            stats["mean"] = stat.get("mean")
            stats["min"] = stat.get("min")
            stats["max"] = stat.get("max")
            stats["std_dev"] = stat.get("standardDeviation")

        logger.info(f"Fetched stats for {city}: AQI mean={stats.get('mean', 'N/A')}")
        return stats

    except requests.RequestException as e:
        logger.error(f"Failed to fetch stats for {city}: {e}")
        return None


def fetch_all_city_stats() -> list[dict[str, Any]]:
    """
    Fetch air quality statistics for all monitored cities.

    Returns:
        List of statistics dictionaries for each city.
    """
    all_stats = []

    for city, bbox in CITY_BBOXES.items():
        stats = fetch_city_stats(city, bbox)
        if stats:
            all_stats.append(stats)

    return all_stats


def format_stats_for_post(stats: list[dict[str, Any]]) -> str:
    """
    Format city statistics into a readable post text.

    Args:
        stats: List of city statistics dictionaries.

    Returns:
        Formatted text suitable for social media posting.
    """
    today = dt.now().strftime("%d/%m/%Y")

    lines = [
        f"🌬️ Qualità dell'Aria - Pianura Padana",
        f"📅 {today}",
        "",
        "📊 Indice di Qualità dell'Aria per città:",
        "",
    ]

    # Sort cities by AQI (worst first)
    sorted_stats = sorted(
        stats,
        key=lambda x: x.get("mean", 0) if x.get("mean") is not None else 0,
        reverse=True,
    )

    for stat in sorted_stats:
        city = stat["city"]
        mean_aqi = stat.get("mean")

        if mean_aqi is not None:
            # Round to get AQI level
            aqi_level = min(6, max(1, round(mean_aqi)))
            level_name, emoji = AQI_LEVELS.get(aqi_level, ("Unknown", "❓"))
            lines.append(f"{emoji} {city}: {level_name} ({mean_aqi:.1f})")
        else:
            lines.append(f"❓ {city}: Dati non disponibili")

    lines.extend([
        "",
        "🔗 Fonte: European Environment Agency",
        "",
        "#AirQuality #QualitaDellAria #PianuraPadana #Italia #Smog #AQI",
    ])

    return "\n".join(lines)


def post_to_instagram(image_url: str, caption: str) -> bool:
    """
    Post an image with caption to Instagram using the Graph API.

    Requires environment variables:
    - INSTAGRAM_ACCESS_TOKEN: Long-lived access token
    - INSTAGRAM_BUSINESS_ACCOUNT_ID: Instagram Business Account ID

    Args:
        image_url: Public URL of the image (required by Instagram API).
        caption: Caption text for the post.

    Returns:
        True if post was successful, False otherwise.
    """
    access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
    account_id = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID")

    if not access_token or not account_id:
        logger.error("Instagram credentials not configured. Set INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_BUSINESS_ACCOUNT_ID")
        return False

    if not image_url:
        logger.error("No image URL provided. Instagram requires a publicly accessible image URL.")
        return False

    try:
        # Step 1: Create a media container
        container_url = f"https://graph.facebook.com/v18.0/{account_id}/media"
        container_params = {
            "image_url": image_url,
            "caption": caption,
            "access_token": access_token,
        }

        container_response = requests.post(container_url, data=container_params, timeout=60)
        container_response.raise_for_status()
        container_data = container_response.json()

        if "id" not in container_data:
            logger.error(f"Failed to create media container: {container_data}")
            return False

        container_id = container_data["id"]
        logger.info(f"Created media container: {container_id}")

        # Step 2: Publish the media container
        publish_url = f"https://graph.facebook.com/v18.0/{account_id}/media_publish"
        publish_params = {
            "creation_id": container_id,
            "access_token": access_token,
        }

        publish_response = requests.post(publish_url, data=publish_params, timeout=60)
        publish_response.raise_for_status()
        publish_data = publish_response.json()

        if "id" in publish_data:
            logger.info(f"Successfully posted to Instagram! Post ID: {publish_data['id']}")
            return True
        else:
            logger.error(f"Failed to publish to Instagram: {publish_data}")
            return False

    except requests.RequestException as e:
        logger.error(f"Instagram API error: {e}")
        return False


def _main():
    """Main function to fetch air quality data and save to files."""
    logger.info("Starting Aria Padana air quality update...")

    # Ensure output directory exists
    os.makedirs("./out", exist_ok=True)

    # Fetch the air quality image
    image_path = fetch_air_quality_image("./out/aria_padana.png")
    if not image_path:
        logger.error("Failed to fetch air quality image")
        return

    # Fetch statistics for all cities
    stats = fetch_all_city_stats()
    if not stats:
        logger.error("Failed to fetch any city statistics")
        return

    # Add legend and city labels to the image
    add_legend_to_image(image_path, city_stats=stats)

    # Format the caption
    caption = format_stats_for_post(stats)
    logger.info(f"Generated caption:\n{caption}")

    # Save caption and stats to files
    output_data = {
        "caption": caption,
        "stats": stats,
        "image_path": image_path,
        "generated_at": dt.now().isoformat(),
    }

    with open("./out/aria_padana.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("Saved output to ./out/aria_padana.json")
    logger.info(f"Image saved at: {image_path}")


def _post():
    """Post to Instagram using pre-generated data."""
    logger.info("Starting Instagram post...")

    # Load the generated data
    try:
        with open("./out/aria_padana.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error("No data found. Run labnotes-aria-padana first to generate data.")
        return

    caption = data["caption"]

    # Get image URL (set after S3 upload)
    image_url = os.getenv("ARIA_PADANA_IMAGE_URL")
    if not image_url:
        logger.error("ARIA_PADANA_IMAGE_URL not set. Upload image to S3 first.")
        return

    # Post to Instagram
    success = post_to_instagram(image_url, caption)
    if success:
        logger.info("Successfully posted to Instagram")
    else:
        logger.error("Failed to post to Instagram")


def main():
    """CLI entry point for fetching data and generating files."""
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


def post_main():
    """CLI entry point for posting to Instagram."""
    try:
        setup_logging()
        _post()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Unexpected error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
