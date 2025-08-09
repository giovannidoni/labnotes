#!/usr/bin/env python3
"""
Convert OPML to feeds.yaml structure.
"""
import argparse, xml.etree.ElementTree as ET, yaml

def parse_opml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    body = root.find('body')
    groups = {}
    for group in body.findall('outline'):
        title = group.attrib.get('text') or group.attrib.get('title') or 'Feeds'
        urls = []
        for child in group.findall('outline'):
            url = child.attrib.get('xmlUrl')
            if url:
                urls.append(url)
        if urls:
            groups[title] = urls
    return groups

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opml", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data = parse_opml(args.opml)
    with open(args.out, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Wrote {args.out} with {sum(len(v) for v in data.values())} feeds.")

if __name__ == "__main__":
    main()
