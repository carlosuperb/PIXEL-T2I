"""
t2_build_4view_from_sheet_demo.py

This script extracts the four static standing poses (north, west, south, east)
from full LPC-style spritesheets and arranges them into a 2x2 grid:

    [ north | west ]
    [ south | east ]

It processes two demo sheets:

    character_full_sheet_demo1.png -> character_4view_demo1.png
    character_full_sheet_demo2.png -> character_4view_demo2.png
"""

from pathlib import Path
from PIL import Image

# Standard LPC tile size
TILE_W = 64
TILE_H = 64

# Standing-frame rows in the assembled sheets
COORDS = {
    "north": (8, 0),
    "west":  (9, 0),
    "south": (10, 0),
    "east":  (11, 0),
}

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

DATASET_ROOT = ROOT / "pixel_character_dataset"
DEMO_DIR = DATASET_ROOT / "processed" / "demo"


def cut_tile(sheet: Image.Image, row: int, col: int) -> Image.Image:
    """Extract a 64x64 tile at (row, col)."""
    left = col * TILE_W
    upper = row * TILE_H
    right = left + TILE_W
    lower = upper + TILE_H
    return sheet.crop((left, upper, right, lower))


def build_4view_for_sheet(sheet_path: Path, out_path: Path) -> None:
    """Build a 2x2 4-view image from one full-character sheet."""
    sheet = Image.open(sheet_path).convert("RGBA")

    # Extract frames
    north = cut_tile(sheet, *COORDS["north"])
    west  = cut_tile(sheet, *COORDS["west"])
    south = cut_tile(sheet, *COORDS["south"])
    east  = cut_tile(sheet, *COORDS["east"])

    # Create 2x2 canvas
    canvas = Image.new("RGBA", (TILE_W * 2, TILE_H * 2))

    # Layout:
    # [ north | west ]
    # [ south | east ]
    canvas.paste(north, (0, 0))
    canvas.paste(west,  (TILE_W, 0))
    canvas.paste(south, (0, TILE_H))
    canvas.paste(east,  (TILE_W, TILE_H))

    canvas.save(out_path)
    print(f"Saved 4-view to: {out_path}")


def main():
    # Sheet 1 -> 4view demo1
    sheet1 = DEMO_DIR / "character_full_sheet_demo1.png"
    out1   = DEMO_DIR / "character_4view_demo1.png"
    build_4view_for_sheet(sheet1, out1)

    # Sheet 2 -> 4view demo2
    sheet2 = DEMO_DIR / "character_full_sheet_demo2.png"
    out2   = DEMO_DIR / "character_4view_demo2.png"
    build_4view_for_sheet(sheet2, out2)


if __name__ == "__main__":
    main()
