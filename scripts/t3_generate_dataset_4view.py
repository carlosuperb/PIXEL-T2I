"""
t3_generate_dataset_4view.py

This script generates a dataset of 2x2 four-view character sprites from the
assembled LPC-style layered assets, together with natural-language captions.

It randomly samples combinations of layers (body, legs, torso, hair, and 
optional equipment layers), composites the tiles for four directions using
the row indices of the assembled sheet:

    north → row 8
    west  → row 9
    south → row 10
    east  → row 11

The final 2x2 layout follows the clockwise 4-view format:

    [ north | west ]
    [ south | east ]

For each sampled combination, the script outputs:

- PNG images to: pixel_character_dataset/processed/dataset_4view/images
- a CSV file containing: image_path, caption

Captions are constructed from the selected asset paths using 
build_caption_from_combo() in t4_lpc_caption_utils.py.
"""

from pathlib import Path
from PIL import Image
import random
import csv

from t4_lpc_caption_utils import build_caption_from_combo


# ================== BASIC SETTINGS ==================
TILE_W = 64
TILE_H = 64

# Direction → row index in the assembled LPC spritesheet
ROW_MAP = {
    "north": 8,   # back
    "west":  9,   # left
    "south": 10,  # front
    "east":  11,  # right
}

# Always using the first column of each row
COL_IDX = 0

# Number of samples to generate
NUM_SAMPLES = 50000   # TODO: increase to 50000 to match 4view dataset


# ================== PATH SETTINGS ==================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

DATASET_ROOT = ROOT / "pixel_character_dataset"
ASSET_ROOT   = DATASET_ROOT / "raw_assets" / "Universal-LPC-spritesheet"

IMG_OUT_DIR  = DATASET_ROOT / "processed" / "dataset_4view" / "images"
CAPTION_CSV  = DATASET_ROOT / "processed" / "dataset_4view" / "captions.csv"

IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)

print("ASSET_ROOT:", ASSET_ROOT)
print("IMG_OUT_DIR:", IMG_OUT_DIR)


# ================== LAYER OPTIONS (auto-discovery) ==================
# NOTE:
# - All paths are stored *relative* to ASSET_ROOT.
# - For body: only PNGs directly under body/male and body/female are used;
#   subfolders like ears/eyes/nose are ignored.
# - For other layers (hair/torso/legs/etc.), this script automatically scans
#   the corresponding asset directories and collects suitable PNGs.


def collect_body_options() -> list[str]:
    """Collect base body sprites (exclude ears/eyes/nose)."""
    results: list[str] = []
    for gender in ("male", "female"):
        gdir = ASSET_ROOT / "body" / gender
        if not gdir.exists():
            continue
        # Only PNG files directly under body/male or body/female
        for p in gdir.glob("*.png"):
            rel = p.relative_to(ASSET_ROOT).as_posix()
            results.append(rel)
    return results


def collect_hair_options() -> list[str]:
    """
    Collect hair sprites for both male and female.

    Includes only:
      - subfolders in hair/<gender>/<style>/*.png
        (each style folder contains multiple colour variants)
      - style folders whose names do NOT end with a digit (e.g. skip 'page2')
    """
    results: list[str] = []
    hair_root = ASSET_ROOT / "hair"

    for gender in ("male", "female"):
        gdir = hair_root / gender
        if not gdir.exists():
            continue

        # Colour variants inside each style folder
        for style_dir in gdir.iterdir():
            if not style_dir.is_dir():
                continue
            
            style_name = style_dir.name.lower()
            # skip duplicated variants like bangslong2, messy2, page2, ponytail2, etc.
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
    Collect torso (upper clothing / armor / capes / wings) sprites.

    Includes any .png under torso/, excluding mask/template/incomplete helper files.
    """
    results: list[str] = []
    torso_root = ASSET_ROOT / "torso"

    if not torso_root.exists():
        return results

    # Traverse recursively: torso/**/*.png
    for p in torso_root.rglob("*.png"):
        rel = p.relative_to(ASSET_ROOT).as_posix()
        lower_rel = rel.lower()
        name = p.name.lower()

        # Skip helper / WIP files
        if (
            "mask" in name
            or "template" in name
            or "incomplete" in name
            or "spikes" in name
        ):
            continue

        # Accept everything else, including torso/back/*
        results.append(rel)

    return results


def collect_legs_options() -> list[str]:
    """
    Collect leg sprites (armor, pants, skirt) for both male and female.
    """
    results: list[str] = []
    legs_root = ASSET_ROOT / "legs"

    # Loop over armor / pants / skirt and male / female
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

                # Skip incomplete or WIP assets
                if "incomplete" in name:
                    continue

                rel = p.relative_to(ASSET_ROOT).as_posix()
                results.append(rel)

    return results


def collect_headgear_options() -> list[str]:
    """
    Collect headgear sprites (bandanas, caps, helms, hoods, tiaras).

    Rules:
      - include any *.png inside <category>/<gender>/ subfolders
      - include *.png under tiaras_female/
      - skip files whose names contain mask/template/incomplete
    """
    results: list[str] = []
    head_root = ASSET_ROOT / "head"

    if not head_root.exists():
        return results

    for category_dir in head_root.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name.lower()

        # Special case: tiaras_female (no /male/ folder)
        if category_name == "tiaras_female":
            for p in category_dir.glob("*.png"):
                name = p.name.lower()
                if any(x in name for x in ("mask", "template", "incomplete")):
                    continue
                rel = p.relative_to(ASSET_ROOT).as_posix()
                results.append(rel)
            continue

        # Normal case: category/<gender>/*.png
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
    """
    Collect weapon sprites.

    Rules:
      - include *.png under weapons/
      - skip oversize folders
      - skip certain unwanted folders
      - skip specific unwanted filenames
      - skip files whose names contain mask/template/incomplete
    """
    results: list[str] = []
    weapons_root = ASSET_ROOT / "weapons"

    if not weapons_root.exists():
        return results

    # folders to skip
    SKIP_FOLDERS = [
        "both hand",
        "either",              # covers left hand/either AND right hand/either
    ]

    # files to skip by name
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
        "spear.png",  # both-hand spear
    ]

    for p in weapons_root.rglob("*.png"):
        rel = p.relative_to(ASSET_ROOT).as_posix()
        lower_rel = rel.lower()
        name = p.name.lower()

        # Skip oversize
        if "oversize" in lower_rel:
            continue

        # Skip helper/template files
        if any(x in name for x in ("mask", "template", "incomplete")):
            continue

        # Skip unwanted folders
        if any(folder in lower_rel for folder in SKIP_FOLDERS):
            continue

        # Skip unwanted files
        if name in SKIP_FILES:
            continue

        results.append(rel)

    return results


def collect_hands_options() -> list[str]:
    """
    Collect hand-layer sprites (bracelets, bracers, gloves).
    """
    results: list[str] = []
    hands_root = ASSET_ROOT / "hands"

    if not hands_root.exists():
        return results

    # Recursive search: hands/**\/*.png
    for p in hands_root.rglob("*.png"):
        name = p.name.lower()
        rel = p.relative_to(ASSET_ROOT).as_posix()

        # Exclude mask/template/incomplete
        if any(x in name for x in ("mask", "template", "incomplete")):
            continue

        results.append(rel)

    return results


def collect_feet_options() -> list[str]:
    """
    Collect feet-layer sprites (boots, shoes, slippers, ghillies, etc.).
    """
    results: list[str] = []
    feet_root = ASSET_ROOT / "feet"

    if not feet_root.exists():
        return results

    # recursive search: feet/**\/*.png
    for p in feet_root.rglob("*.png"):
        name = p.name.lower()
        rel = p.relative_to(ASSET_ROOT).as_posix()

        # Exclude mask/template/incomplete
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

    # optional layers
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


def pick_combo(rng: random.Random) -> dict:
    """
    Select one option from each layer category.

    Required layers are always present.
    Optional layers may be skipped based on predefined probabilities.
    """
    combo: dict = {}

    for key, options in LAYER_CHOICES.items():
        if key not in REQUIRED_LAYERS:
            # Optional layer probabilities (tune as desired)
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
            # Required layer: always pick one
            combo[key] = rng.choice(options)

    return combo


def compose_direction_frame(combo: dict, dir_key: str) -> Image.Image:
    """
    Compose a single direction frame (one 64x64 tile) by stacking
    all relevant layers for the specified direction.
    """
    row = ROW_MAP[dir_key]
    col = COL_IDX

    canvas = Image.new("RGBA", (TILE_W, TILE_H))
    for layer_name in LAYER_ORDER:
        rel_path = combo.get(layer_name)
        if not rel_path:
            continue  # skip optional layers that are None

        sheet = load_image(rel_path)
        tile = crop_tile(sheet, row, col)
        canvas.alpha_composite(tile)
    return canvas


def compose_4view_image(combo: dict) -> Image.Image:
    """
    Compose a 2x2 four-view sprite (north, west, south, east)
    from the selected layer combination, arranged as:

        [ north | west ]
        [ south | east ]
    """
    north = compose_direction_frame(combo, "north")
    west  = compose_direction_frame(combo, "west")
    south = compose_direction_frame(combo, "south")
    east  = compose_direction_frame(combo, "east")

    canvas = Image.new("RGBA", (TILE_W * 2, TILE_H * 2))

    # Layout:
    # [ north | west ]
    # [ south | east ]
    canvas.paste(north, (0, 0))
    canvas.paste(west,  (TILE_W, 0))
    canvas.paste(south, (0, TILE_H))
    canvas.paste(east,  (TILE_W, TILE_H))

    return canvas


# ================== MAIN DATASET GENERATION LOOP ==================
def main():
    rng = random.Random(42)
    rows = []

    for idx in range(NUM_SAMPLES):
        combo = pick_combo(rng)
        img = compose_4view_image(combo)

        out_path = IMG_OUT_DIR / f"char_{idx:05d}.png"
        img.save(out_path)

        caption = build_caption_from_combo(combo)
        rows.append([out_path.name, caption])

        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{NUM_SAMPLES}] saved:", out_path.name)

    # Write captions CSV
    CAPTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    with CAPTION_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text"])
        writer.writerows(rows)

    print("Done. Captions saved to:", CAPTION_CSV)


if __name__ == "__main__":
    main()
