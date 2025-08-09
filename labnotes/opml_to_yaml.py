import argparse, xml.etree.ElementTree as ET, yaml


def parse_opml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    body = root.find("body")
    groups = {}
    for group in body.findall("outline"):
        title = group.attrib.get("text") or group.attrib.get("title") or "Feeds"
        urls = []
        for child in group.findall("outline"):
            url = child.attrib.get("xmlUrl")
            if url:
                urls.append(url)
        if urls:
            groups[title] = urls
    return groups


def convert_opml_to_yaml(opml_path, output_path):
    """Convert OPML file to YAML format."""
    data = parse_opml(opml_path)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Wrote {output_path} with {sum(len(v) for v in data.values())} feeds.")
