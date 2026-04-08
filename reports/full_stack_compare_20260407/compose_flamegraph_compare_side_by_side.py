#!/usr/bin/env python3

from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None


def load_font(size: int):
    for path in [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: compose_flamegraph_compare_side_by_side.py VLLM.png RVLLM.png OUT.png",
            file=sys.stderr,
        )
        return 2

    left_path, right_path, out_path = map(Path, sys.argv[1:4])
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    top_pad = 96
    bottom_pad = 24
    side_pad = 28
    gap = 40

    width = side_pad + left.width + gap + right.width + side_pad
    height = top_pad + max(left.height, right.height) + bottom_pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(52)
    subtitle_font = load_font(24)

    draw.text((side_pad, 24), "vLLM vs rvLLM flamegraph comparison", fill="black", font=title_font)
    draw.text(
        (side_pad, 68),
        "vLLM left, rvLLM right. Both rendered at 10000 px tall from the original flamegraph sources.",
        fill=(70, 70, 70),
        font=subtitle_font,
    )

    left_x = side_pad
    right_x = side_pad + left.width + gap
    y = top_pad

    canvas.paste(left, (left_x, y))
    canvas.paste(right, (right_x, y))

    label_font = load_font(28)
    draw.text((left_x + 12, y + 28), "vLLM", fill="black", font=label_font)
    draw.text((right_x + 12, y + 28), "rvLLM", fill="black", font=label_font)

    canvas.save(out_path, format="PNG", optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
