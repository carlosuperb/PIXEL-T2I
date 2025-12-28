"""
u1_build_actions_from_sheet_demo.py

Build ONE combined action sheet by extracting multiple action blocks from an
assembled LPC-style full sheet, and stacking them vertically.

Input sheet is provided via CLI (terminal), not hard-coded in this file.

Each action block is a 4-row sheet in clockwise 4-view order:
    row 0 → west
    row 1 → east
    row 2 → south
    row 3 → north

Actions included (rows are based on your assembled sheet convention):
  - walk   rows: north=8,  west=9,  south=10, east=11   cols=9
  - thrust rows: north=4,  west=5,  south=6,  east=7    cols=8
  - slash  rows: north=12, west=13, south=14, east=15   cols=6

Output:
  pixel_character_dataset/processed/demo/<out_name>.png
"""

import argparse
from pathlib import Path
from PIL import Image

# Tile size
TILE_W = 64
TILE_H = 64

# Output rows in clockwise 4-view order
DIRECTIONS = ["west", "east", "south", "north"]

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

DATASET_ROOT = ROOT / "pixel_character_dataset"
DEMO_DIR = DATASET_ROOT / "processed" / "demo"


def cut_tile(sheet: Image.Image, row: int, col: int) -> Image.Image:
    """Crop a 64x64 tile at (row, col) from the full spritesheet."""
    left = col * TILE_W
    upper = row * TILE_H
    right = left + TILE_W
    lower = upper + TILE_H
    return sheet.crop((left, upper, right, lower))


def pad_to_width(img: Image.Image, target_w: int) -> Image.Image:
    """Pad an RGBA image to target_w with transparent pixels on the right."""
    w, h = img.size
    if w == target_w:
        return img
    canvas = Image.new("RGBA", (target_w, h))
    canvas.paste(img, (0, 0))
    return canvas


def build_action_block(sheet: Image.Image, coords: dict, num_cols: int) -> Image.Image:
    """Build a 4-row action block (west/east/south/north) with num_cols frames."""
    block = Image.new("RGBA", (TILE_W * num_cols, TILE_H * len(DIRECTIONS)))

    for row_idx, dir_name in enumerate(DIRECTIONS):
        row = coords[dir_name]
        for col in range(num_cols):
            tile = cut_tile(sheet, row, col)
            block.paste(tile, (col * TILE_W, row_idx * TILE_H))

    return block


def action_specs():
    """
    Return action specifications.
    Only the input sheet changes via CLI; row/col conventions stay the same.
    """
    return [
        {
            "name": "walk",
            "cols": 9,
            "coords": {"north": 8, "west": 9, "south": 10, "east": 11},
        },
        {
            "name": "thrust",
            "cols": 8,  # keep consistent with your current u2 script
            "coords": {"north": 4, "west": 5, "south": 6, "east": 7},
        },
        {
            "name": "slash",
            "cols": 6,
            "coords": {"north": 12, "west": 13, "south": 14, "east": 15},
        },
    ]


def validate_rows(sheet: Image.Image, coords: dict) -> bool:
    """Check if all required row indices exist in the sheet."""
    sheet_rows = sheet.size[1] // TILE_H
    needed = [coords[k] for k in ("north", "west", "south", "east")]
    return all(0 <= r < sheet_rows for r in needed)


def main():
    parser = argparse.ArgumentParser(
        description="Build one combined (walk+thrust+slash) action sheet from a full spritesheet."
    )
    parser.add_argument(
        "--sheet",
        type=str,
        required=True,
        help="Input full spritesheet path (e.g., pixel_character_dataset/processed/demo/character_full_sheet_demo1.png)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="character_actions_demo.png",
        help="Output filename (saved under pixel_character_dataset/processed/demo/). Default: character_actions_demo.png",
    )
    args = parser.parse_args()

    sheet_path = Path(args.sheet)
    if not sheet_path.is_absolute():
        sheet_path = (ROOT / sheet_path).resolve()

    out_path = DEMO_DIR / args.out

    # Load sheet
    sheet = Image.open(sheet_path).convert("RGBA")
    sheet_w, sheet_h = sheet.size
    total_cols = sheet_w // TILE_W
    total_rows = sheet_h // TILE_H

    print("Loaded sheet:", sheet_path)
    print("Sheet size:", sheet_w, "x", sheet_h, f"(cols={total_cols}, rows={total_rows})")

    specs = action_specs()

    # unified width by max columns among actions
    max_cols = max(s["cols"] for s in specs)
    target_w = TILE_W * min(total_cols, max_cols)  # also limited by actual sheet width

    blocks = []

    for s in specs:
        if not validate_rows(sheet, s["coords"]):
            print(f"[WARN] Skip '{s['name']}' (required rows out of range for this sheet).")
            continue

        num_cols = min(total_cols, s["cols"])
        # map to direction order expected by block builder
        coords = {
            "north": s["coords"]["north"],
            "west":  s["coords"]["west"],
            "south": s["coords"]["south"],
            "east":  s["coords"]["east"],
        }

        block = build_action_block(sheet, coords, num_cols)
        block = pad_to_width(block, target_w)
        blocks.append((s["name"], block))
        print(f"Built block: {s['name']} cols={num_cols} size={block.size}")

    if not blocks:
        raise RuntimeError("No action blocks were built. Check your sheet rows/structure.")

    # stack vertically
    total_h_out = sum(img.size[1] for _, img in blocks)
    canvas = Image.new("RGBA", (target_w, total_h_out))

    y = 0
    for name, img in blocks:
        canvas.paste(img, (0, y))
        y += img.size[1]

    canvas.save(out_path)
    print("Saved combined actions sheet to:", out_path)


if __name__ == "__main__":
    main()
