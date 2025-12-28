"""
u2_generate_dataset_actions.py

Generate a dataset of combined ACTION sheets (walk + thrust + slash) from
assembled LPC-style layered assets.

This script samples random layer combinations (same idea as t3_generate_dataset_4view.py),
composites the required frames for each action (walk/thrust/slash), builds
4-row action blocks in clockwise 4-view order:

    row 0 → west
    row 1 → east
    row 2 → south
    row 3 → north

and stacks the action blocks vertically into ONE output spritesheet per sample.

Unlike t3_generate_dataset_4view.py:
  - No captions are generated.
  - Output is a single combined action sheet per sample.

Actions included (rows based on your assembled sheet convention):
  - walk   rows: north=8,  west=9,  south=10, east=11   cols=9
  - thrust rows: north=4,  west=5,  south=6,  east=7    cols=8
  - slash  rows: north=12, west=13, south=14, east=15   cols=6

Output:
  pixel_character_dataset/processed/dataset_actions/images/char_00000.png
"""

from pathlib import Path
from PIL import Image
import random


# ================== BASIC SETTINGS ==================
TILE_W = 64
TILE_H = 64

# Output rows in clockwise 4-view order
DIRECTIONS = ["west", "east", "south", "north"]


# ================== DATASET SETTINGS (same as t3) ==================
SEED = 42
NUM_SAMPLES = 50000   # TODO: set to 50000 for real dataset


# ================== PATH SETTINGS ==================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

DATASET_ROOT = ROOT / "pixel_character_dataset"
ASSET_ROOT   = DATASET_ROOT / "raw_assets" / "Universal-LPC-spritesheet"

IMG_OUT_DIR  = DATASET_ROOT / "processed" / "dataset_actions" / "images"
IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)

print("ASSET_ROOT:", ASSET_ROOT)
print("IMG_OUT_DIR:", IMG_OUT_DIR)


# ================== LAYER OPTIONS (auto-discovery) ==================
def collect_body_options() -> list[str]:
    """Collect base body sprites (exclude ears/eyes/nose)."""
    results: list[str] = []
    for gender in ("male", "female"):
        gdir = ASSET_ROOT / "body" / gender
        if not gdir.exists():
            continue
        for p in gdir.glob("*.png"):
            rel = p.relative_to(ASSET_ROOT).as_posix()
            results.append(rel)
    return results


def collect_hair_options() -> list[str]:
    """
    Collect hair sprites for both male and female.

    Includes only:
      - subfolders in hair/<gender>/<style>/*.png
      - style folders whose names do NOT end with a digit (e.g. skip 'page2')
    """
    results: list[str] = []
    hair_root = ASSET_ROOT / "hair"

    for gender in ("male", "female"):
        gdir = hair_root / gender
        if not gdir.exists():
            continue

        for style_dir in gdir.iterdir():
            if not style_dir.is_dir():
                continue

            style_name = style_dir.name.lower()
            if style_name[-1:].isdigit():
                continue

            for p in style_dir.glob("*.png"):
                name = p.name.lower()
                if "mask" in name or "template" in name:
                    continue
                rel = p.relative_to(ASSET_ROOT).as_posix()
                results.append(rel)

    return results


def collect_torso_options() -> list[str]:
    """
    Collect torso sprites.

    Includes any .png under torso/, excluding mask/template/incomplete helper files.
    """
    results: list[str] = []
    torso_root = ASSET_ROOT / "torso"

    if not torso_root.exists():
        return results

    for p in torso_root.rglob("*.png"):
        rel = p.relative_to(ASSET_ROOT).as_posix()
        name = p.name.lower()

        if (
            "mask" in name
            or "template" in name
            or "incomplete" in name
            or "spikes" in name
        ):
            continue

        results.append(rel)

    return results


def collect_legs_options() -> list[str]:
    """Collect leg sprites (armor, pants, skirt) for both male and female."""
    results: list[str] = []
    legs_root = ASSET_ROOT / "legs"

    for category in ("armor", "pants", "skirt"):
        cdir = legs_root / category
        if not cdir.exists():
            continue

        for gender in ("male", "female"):
            gdir = cdir / gender
            if not gdir.exists():
                continue

            for p in gdir.glob("*.png"):
                name = p.name.lower()
                if "incomplete" in name:
                    continue
                rel = p.relative_to(ASSET_ROOT).as_posix()
                results.append(rel)

    return results


def collect_headgear_options() -> list[str]:
    """Collect headgear sprites (bandanas, caps, helms, hoods, tiaras)."""
    results: list[str] = []
    head_root = ASSET_ROOT / "head"

    if not head_root.exists():
        return results

    for category_dir in head_root.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name.lower()

        if category_name == "tiaras_female":
            for p in category_dir.glob("*.png"):
                name = p.name.lower()
                if any(x in name for x in ("mask", "template", "incomplete")):
                    continue
                rel = p.relative_to(ASSET_ROOT).as_posix()
                results.append(rel)
            continue

        for gender in ("male", "female"):
            gdir = category_dir / gender
            if not gdir.exists():
                continue
            for p in gdir.glob("*.png"):
                name = p.name.lower()
                if any(x in name for x in ("mask", "template", "incomplete")):
                    continue
                rel = p.relative_to(ASSET_ROOT).as_posix()
                results.append(rel)

    return results


def collect_weapon_options() -> list[str]:
    """Collect weapon sprites (same rules as t3)."""
    results: list[str] = []
    weapons_root = ASSET_ROOT / "weapons"

    if not weapons_root.exists():
        return results

    SKIP_FOLDERS = [
        "both hand",
        "either",
    ]

    SKIP_FILES = [
        "shield_male_cutoutforhat.png",
        "steelwand_female.png",
        "woodwand_female.png",
        "woodwand_male.png",
        "arrow.png",
        "arrow_skeleton.png",
        "bow.png",
        "bow_skeleton.png",
        "greatbow.png",
        "recurvebow.png",
        "spear.png",
    ]

    for p in weapons_root.rglob("*.png"):
        rel = p.relative_to(ASSET_ROOT).as_posix()
        lower_rel = rel.lower()
        name = p.name.lower()

        if "oversize" in lower_rel:
            continue

        if any(x in name for x in ("mask", "template", "incomplete")):
            continue

        if any(folder in lower_rel for folder in SKIP_FOLDERS):
            continue

        if name in SKIP_FILES:
            continue

        results.append(rel)

    return results


def collect_hands_options() -> list[str]:
    """Collect hand-layer sprites (bracelets, bracers, gloves)."""
    results: list[str] = []
    hands_root = ASSET_ROOT / "hands"

    if not hands_root.exists():
        return results

    for p in hands_root.rglob("*.png"):
        name = p.name.lower()
        rel = p.relative_to(ASSET_ROOT).as_posix()
        if any(x in name for x in ("mask", "template", "incomplete")):
            continue
        results.append(rel)

    return results


def collect_feet_options() -> list[str]:
    """Collect feet-layer sprites (boots, shoes, slippers, ghillies, etc.)."""
    results: list[str] = []
    feet_root = ASSET_ROOT / "feet"

    if not feet_root.exists():
        return results

    for p in feet_root.rglob("*.png"):
        name = p.name.lower()
        rel = p.relative_to(ASSET_ROOT).as_posix()
        if any(x in name for x in ("mask", "template", "incomplete")):
            continue
        results.append(rel)

    return results


REQUIRED_LAYERS = ["body", "legs", "torso", "hair"]
OPTIONAL_LAYERS = ["headgear", "weapons", "hands", "feet"]

LAYER_CHOICES = {
    "body": collect_body_options(),
    "hair": collect_hair_options(),
    "torso": collect_torso_options(),
    "legs": collect_legs_options(),

    "headgear": collect_headgear_options(),
    "weapons": collect_weapon_options(),
    "hands": collect_hands_options(),
    "feet": collect_feet_options(),
}

# Layer blending order: from background to foreground
LAYER_ORDER = [
    "body",
    "legs",
    "feet",
    "torso",
    "hands",
    "hair",
    "headgear",
    "weapons",
]


# ================== ACTION SPECS (same as u1) ==================
def action_specs():
    """Return action specifications (fixed conventions)."""
    return [
        {
            "name": "walk",
            "cols": 9,
            "coords": {"north": 8, "west": 9, "south": 10, "east": 11},
        },
        {
            "name": "thrust",
            "cols": 8,
            "coords": {"north": 4, "west": 5, "south": 6, "east": 7},
        },
        {
            "name": "slash",
            "cols": 6,
            "coords": {"north": 12, "west": 13, "south": 14, "east": 15},
        },
    ]


# ================== UTILITY FUNCTIONS ==================
def load_image(rel_path: str) -> Image.Image:
    """Load a full spritesheet layer from relative path."""
    path = ASSET_ROOT / rel_path
    return Image.open(path).convert("RGBA")


def crop_tile(sheet: Image.Image, row: int, col: int) -> Image.Image:
    """Crop a single TILE_W x TILE_H tile from a spritesheet."""
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


def validate_rows(sheet_rows: int, coords: dict) -> bool:
    """Check if all required row indices exist in the sheet."""
    needed = [coords[k] for k in ("north", "west", "south", "east")]
    return all(0 <= r < sheet_rows for r in needed)


def pick_combo(rng: random.Random) -> dict:
    """
    Select one option from each layer category.

    Required layers are always present.
    Optional layers may be skipped based on predefined probabilities.
    """
    combo: dict = {}

    for key, options in LAYER_CHOICES.items():
        if not options:
            combo[key] = None
            continue

        if key not in REQUIRED_LAYERS:
            if key == "headgear" and rng.random() < 0.2:
                combo[key] = rng.choice(options)
            elif key == "weapons" and rng.random() < 0.3:
                combo[key] = rng.choice(options)
            elif key == "hands" and rng.random() < 0.6:
                combo[key] = rng.choice(options)
            elif key == "feet" and rng.random() < 0.7:
                combo[key] = rng.choice(options)
            else:
                combo[key] = None
        else:
            combo[key] = rng.choice(options)

    return combo


def compose_tile_from_layers(layer_sheets: dict, row: int, col: int) -> Image.Image:
    """Composite one 64x64 tile from preloaded layer sheets at (row, col)."""
    canvas = Image.new("RGBA", (TILE_W, TILE_H))
    for layer_name in LAYER_ORDER:
        sheet = layer_sheets.get(layer_name)
        if sheet is None:
            continue
        tile = crop_tile(sheet, row, col)
        canvas.alpha_composite(tile)
    return canvas


def build_action_block(layer_sheets: dict, coords: dict, num_cols: int) -> Image.Image:
    """Build a 4-row action block (west/east/south/north) with num_cols frames."""
    block = Image.new("RGBA", (TILE_W * num_cols, TILE_H * len(DIRECTIONS)))

    for row_idx, dir_name in enumerate(DIRECTIONS):
        row = coords[dir_name]  # coords uses {west/east/south/north}
        for col in range(num_cols):
            tile = compose_tile_from_layers(layer_sheets, row, col)
            block.paste(tile, (col * TILE_W, row_idx * TILE_H))

    return block


def build_combined_actions_sheet(combo: dict) -> Image.Image:
    """Build ONE combined action sheet (walk+thrust+slash stacked vertically)."""
    # preload selected layer sheets once per sample
    layer_sheets: dict = {}
    for layer_name in LAYER_ORDER:
        rel_path = combo.get(layer_name)
        if not rel_path:
            layer_sheets[layer_name] = None
            continue
        layer_sheets[layer_name] = load_image(rel_path)

    # infer sheet cols/rows from any existing sheet
    ref = None
    for v in layer_sheets.values():
        if v is not None:
            ref = v
            break
    if ref is None:
        raise RuntimeError("No layers loaded (unexpected).")

    sheet_w, sheet_h = ref.size
    total_cols = sheet_w // TILE_W
    total_rows = sheet_h // TILE_H

    specs = action_specs()

    # unified width by max columns among actions (also limited by actual sheet width)
    max_cols = max(s["cols"] for s in specs)
    target_w = TILE_W * min(total_cols, max_cols)

    blocks = []

    for s in specs:
        if not validate_rows(total_rows, s["coords"]):
            print(f"[WARN] Skip '{s['name']}' (required rows out of range for this sheet).")
            continue

        num_cols = min(total_cols, s["cols"])

        # map to direction order expected by builder: west/east/south/north
        coords = {
            "west":  s["coords"]["west"],
            "east":  s["coords"]["east"],
            "south": s["coords"]["south"],
            "north": s["coords"]["north"],
        }

        block = build_action_block(layer_sheets, coords, num_cols)
        block = pad_to_width(block, target_w)
        blocks.append((s["name"], block))

    if not blocks:
        raise RuntimeError("No action blocks were built. Check your sheet rows/structure.")

    # stack vertically
    total_h_out = sum(img.size[1] for _, img in blocks)
    canvas = Image.new("RGBA", (target_w, total_h_out))

    y = 0
    for name, img in blocks:
        canvas.paste(img, (0, y))
        y += img.size[1]

    return canvas


# ================== MAIN DATASET GENERATION LOOP ==================
def main():
    rng = random.Random(SEED)

    for idx in range(NUM_SAMPLES):
        combo = pick_combo(rng)
        img = build_combined_actions_sheet(combo)

        out_path = IMG_OUT_DIR / f"char_{idx:05d}.png"
        img.save(out_path)

        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{NUM_SAMPLES}] saved:", out_path.name)

    print("Done. Images saved to:", IMG_OUT_DIR)


if __name__ == "__main__":
    main()
