#!/usr/bin/env python3

import sys
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def q(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def parse_num(value: str) -> float:
    value = value.strip()
    if value.endswith("%"):
        return float(value[:-1])
    return float(value)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: make_flagged_flamegraph.py INPUT.svg OUTPUT.svg", file=sys.stderr)
        return 2

    src, dst = sys.argv[1], sys.argv[2]
    tree = ET.parse(src)
    root = tree.getroot()
    frames = root.find(f".//{q('svg')}[@id='frames']")
    if frames is None:
        raise SystemExit("could not find frames svg")

    orig_width = int(float(root.attrib["width"]))
    orig_height = int(float(root.attrib["height"]))
    legend_width = 2200
    new_width = orig_width + legend_width
    legend_x = orig_width + 30

    labeled = []
    for g in frames.findall(q("g")):
        text = g.find(q("text"))
        rect = g.find(q("rect"))
        title = g.find(q("title"))
        if text is None or rect is None or title is None:
            continue
        if not (text.text or "").strip():
            continue
        labeled.append(
            {
                "g": g,
                "text": text,
                "title": title.text or "",
                "y": parse_num(rect.attrib["y"]),
                "x": parse_num(rect.attrib["x"]),
            }
        )

    labeled.sort(key=lambda item: (item["y"], item["x"], item["title"]))

    root.set("width", str(new_width))
    root.set("viewBox", f"0 0 {new_width} {orig_height}")

    bg = ET.Element(
        q("rect"),
        {
            "x": str(orig_width),
            "y": "0",
            "width": str(legend_width),
            "height": str(orig_height),
            "fill": "rgb(252,252,248)",
        },
    )
    divider = ET.Element(
        q("line"),
        {
            "x1": str(orig_width),
            "x2": str(orig_width),
            "y1": "0",
            "y2": str(orig_height),
            "stroke": "rgb(180,180,180)",
            "stroke-width": "2",
        },
    )
    root.insert(len(root) - 1, bg)
    root.insert(len(root) - 1, divider)

    header = ET.Element(
        q("text"),
        {
            "x": str(legend_x),
            "y": "28",
            "fill": "rgb(0,0,0)",
            "style": "font-family:monospace;font-size:18px;font-weight:bold",
        },
    )
    header.text = "Flagged labels"
    root.insert(len(root) - 1, header)

    sub = ET.Element(
        q("text"),
        {
            "x": str(legend_x),
            "y": "48",
            "fill": "rgb(70,70,70)",
            "style": "font-family:monospace;font-size:11px",
        },
    )
    sub.text = "Each [n] marker on the flamegraph maps to the full label here."
    root.insert(len(root) - 1, sub)

    line_height = max(11, int((orig_height - 80) / max(1, len(labeled))))
    font_size = max(7, min(11, line_height - 1))
    y = 72

    for i, item in enumerate(labeled, start=1):
        item["text"].text = f"[{i}]"
        line = ET.Element(
            q("text"),
            {
                "x": str(legend_x),
                "y": str(y),
                "fill": "rgb(0,0,0)",
                "style": f"font-family:monospace;font-size:{font_size}px",
            },
        )
        line.text = f"[{i}] {item['title']}"
        root.insert(len(root) - 1, line)
        y += line_height

    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
