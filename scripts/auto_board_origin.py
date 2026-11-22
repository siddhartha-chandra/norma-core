#!/usr/bin/env python3
"""Add a consistent board origin pixel to every hand-eye sample, without clicking.

With an even x odd inner-corner grid, the two diagonal end corners of the board
touch squares of opposite colour. The origin is the end whose outer square is dark.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from measure_coin import DEFAULT_CALIBRATION, board_correspondences


def outer_square_mean(gray, corner, diagonal_neighbour):
    centre = corner + 0.5 * (corner - diagonal_neighbour)
    x, y = np.round(centre).astype(int)
    radius = max(2, int(0.15 * np.linalg.norm(corner - diagonal_neighbour)))
    patch = gray[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
    return float(patch.mean()) if patch.size else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    board = json.loads(args.calibration.read_text())["wrist"]["board"]
    columns, rows = board["inner_columns"], board["inner_rows"]
    if (columns + rows) % 2 == 0:
        sys.exit("Board corners are colour-symmetric; the origin must be clicked.")
    samples = json.loads(args.samples.read_text())
    for sample in samples:
        image = cv2.imread(str(args.samples.parent / sample["image"]))
        try:
            _, pixels, _ = board_correspondences(image, board, False, interactive_fallback=False)
        except ValueError as error:
            print(f"{sample['image']}: {error}", file=sys.stderr)
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        first = outer_square_mean(gray, pixels[0], pixels[columns + 1])
        last = outer_square_mean(gray, pixels[-1], pixels[-(columns + 2)])
        origin = pixels[0] if first < last else pixels[-1]
        sample["origin_pixel"] = origin.tolist()
        print(f"{sample['image']}: dark={min(first, last):.0f} light={max(first, last):.0f}")
    args.output.write_text(json.dumps(samples, indent=2) + "\n")


if __name__ == "__main__":
    main()
