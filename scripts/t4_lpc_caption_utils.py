"""
t4_lpc_caption_utils.py

Utilities for converting LPC asset relative paths into natural-language phrases.
Used for building training captions for diffusion / LoRA models.

Note:
- The functions in this module are intended to be imported.
- The __main__ block at the bottom is only for local debugging / verification.
"""

import os
import re
from typing import List
from pathlib import Path

# Local paths (only used by the optional __main__ debug run)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATASET_ROOT = ROOT / "pixel_character_dataset"
ASSET_ROOT = DATASET_ROOT / "raw_assets" / "Universal-LPC-spritesheet"

# Debug-only: choose one asset category to print parsed descriptions for
DIR_NAME = "head"

# ------------------- base color extraction -------------------
def _extract_color_from_name(name: str) -> str | None:
    """
    Extract simplified color names for non-hair assets (torsos, capes, armor, robes).
    Hair-specific colors are intentionally excluded.
    """
    # only last component, no extension
    s = name.lower().split("/")[-1]
    s = re.sub(r"\.[a-z0-9]+$", "", s)
    s = s.replace("_", "-").replace(" ", "-")  # unify separators
    s = re.sub(r"\d+", "", s)

    SIMPLE_COLORS = {
        "black": "black",
        "white": "white",
        "gray": "gray",
        "grey": "gray",

        "brown": "brown",
        "maroon": "maroon",
        "lavender": "lavender",

        "green": "green",
        "blue": "blue",
        "purple": "purple",
        "red": "red",
        "magenta": "magenta",
        "pink": "pink",
        "yellow": "yellow",
        "teal": "teal",
        "cyan": "cyan",
    }

    # match by substring (simple, stable)
    for key, value in SIMPLE_COLORS.items():
        if key in s:
            return value

    return None


# ------------------- material extraction -------------------
def _extract_material_from_name(name: str) -> str | None:
    """
    Extract material names for equipment-like items (headgear, armor, accessories).
    Examples: gold, silver, bronze, iron, steel, chain, leather, cloth, metal.
    """
    s = name.lower()
    s = re.sub(r"\.[a-z0-9]+$", "", s)
    s = s.replace("_", "-").replace(" ", "-")
    s = re.sub(r"\d+", "", s)

    MATERIALS = {
        # metals
        "golden": "gold",
        "gold": "gold",
        "silver": "silver",
        "bronze": "bronze",
        "iron": "iron",
        "steel": "steel",
        "plate": "steel",
        "metal": "metal",

        # chain, leather, cloth
        "chain": "chain",
        "leather": "leather",
        "cloth": "cloth",

        # wands
        "wood": "wood",
    }

    for key, value in MATERIALS.items():
        if key in s:
            return value

    return None


# ------------------- combine material or color -------------------
def _apply_material_color(base_type: str, filename: str, path: str) -> List[str]:
    """
    Combine material or color with a base type.
    Priority:
      1. material from filename/path
      2. color from filename/path
      3. plain base_type
    Returns list[str] to match describe_* behavior.
    """
    # material first
    material = (
        _extract_material_from_name(filename)
        or _extract_material_from_name(path)
    )

    if material:
        return [f"{material} {base_type}"]

    # color fallback
    color = (
        _extract_color_from_name(filename)
        or _extract_color_from_name(path)
    )

    if color:
        return [f"{color} {base_type}"]

    # no material or color
    return [base_type]


# -------------------- skin tone mapping --------------------
SKIN_TONE_MAP: dict[str, str] = {
    "light": "light skin",
    "tanned": "tanned skin",
    "tanned2": "tanned skin",
    "dark": "dark skin",
    "dark2": "dark skin",
}


# -------------------- body --------------------
def describe_body(rel_path: str) -> List[str]:
    """
    Describe body type (gender template, race, and skin tone).
    Example input: "body/male/light.png"
    """
    desc: List[str] = []
    path = rel_path.lower()

    # --- Gender ---
    if "/male/" in path:
        desc.append("male body template")
    elif "/female/" in path:
        desc.append("female body template")

    # --- Race and unified skin tone rules ---

    # Skeleton: has no skin tone
    if "skeleton" in path:
        desc.append("skeleton")

    # Orc and red_orc share the same base model, treat both as "orc"
    # Both use the unified skin tone "green skin"
    elif "orc" in path:
        desc.append("orc")
        desc.append("green skin")

    # Dark elf (darkelf and darkelf2) use a unified purple skin tone
    elif "darkelf" in path:
        desc.append("dark elf")
        desc.append("purple skin")

    # Human: apply human skin tone mapping
    else:
        desc.append("human")

        # light / tanned / dark skin mapping
        for key, value in SKIN_TONE_MAP.items():
            if key in path:
                desc.append(value)
                break

    return desc


# -------------------- hair --------------------
def _extract_hair_color(name: str) -> str | None:
    """
    Extract hair color name from LPC hair filename.
    Only handles the known LPC hair color set.
    """
    # keep only last component, strip extension & digits, normalize separators
    s = name.lower().split("/")[-1]
    s = re.sub(r"\.[a-z0-9]+$", "", s)   # remove extension
    s = re.sub(r"\d+", "", s)           # blonde2 -> blonde
    s = s.replace("_", "-")             # unify "_" and "-"

    HAIR_COLOR_MAP = {
        # basic single colors
        "black": "black",
        "blonde": "blonde",
        "blue": "blue",
        "brown": "brown",
        "brunette": "brunette",
        "gray": "gray",
        "green": "green",
        "pink": "pink",
        "purple": "purple",
        "white": "white",

        # compound / special hair colors (normalized to space-separated English)
        "dark-blonde": "dark blonde",
        "light-blonde": "light blonde",
        "white-blonde": "white blonde",
        "white-cyan": "white cyan",
        "ruby-red": "ruby red",

        # hair-specific aliases
        "raven": "raven",
        "redhead": "red",
        "gold": "gold",
    }

    return HAIR_COLOR_MAP.get(s, None)


def _normalize_hair_style(style: str) -> str:
    """Map LPC hair folder names to human-readable hairstyle labels."""
    s = style.lower()
    s = re.sub(r"\d+$", "", s)  # messy1 -> messy, page2 -> page

    # bangs group: bangs / bangslong / bangsshort ...
    if s.startswith("bangs"):
        if "short" in s:
            return "short bangs hairstyle"
        if "long" in s:
            return "long bangs hairstyle"
        return "bangs hairstyle"

    # ponytail group: ponytail / ponytail2
    if "ponytail" in s:
        return "ponytail hairstyle"

    # mohawk variants: mohawk / longhawk / shorthawk
    if "longhawk" in s:
        return "long mohawk hairstyle"
    if "shorthawk" in s:
        return "short mohawk hairstyle"
    if "mohawk" in s:
        return "mohawk hairstyle"

    # long hair
    if s == "long":
        return "long hairstyle"

    # extra long hair
    if s == "xlong":
        return "extra-long hairstyle"

    # topknot variants: xlongknot / longknot / shortknot / knot
    if "knot" in s:
        if "xlong" in s:
            return "extra-long topknot hairstyle"
        if "long" in s:
            return "long topknot hairstyle"
        if "short" in s:
            return "short topknot hairstyle"
        return "topknot hairstyle"

    # curly / afro styles
    if "jewfro" in s:
        return "curly afro hairstyle"

    # loose hair
    if "loose" in s:
        return "loose hair"

    # messy / bedhead / unkempt are distinct LPC styles
    if "messy" in s:
        return "messy hairstyle"

    if "bedhead" in s:
        return "bedhead hairstyle"

    if "unkempt" in s:
        return "unkempt hairstyle"

    # pageboy hairstyle
    if "page" in s:
        return "pageboy hairstyle"

    # special named hairstyles
    if "princess" in s:
        return "princess hairstyle"
    if "pixie" in s:
        return "pixie cut hairstyle"
    if "bunches" in s:
        return "twin-tailed hairstyle"
    if "swoop" in s:
        return "swoop hairstyle"

    # shoulder-left and shoulder-right variants
    if "shoulderl" in s:
        return "left-swept hairstyle"
    if "shoulderr" in s:
        return "right-swept hairstyle"

    # generic / fallback short styles
    if "plain" in s:
        return "plain hairstyle"
    if "parted" in s:
        return "parted hairstyle"
    if "short" in s:
        return "short hairstyle"

    return "generic hairstyle"


def describe_hair(rel_path: str) -> List[str]:
    """
    Describe hair style and color.
    Only accepts real hair assets:
        hair/<gender>/<style>/<color>.png
    Skips preview-only files like:
        hair/<gender>/<style>.png
    """
    desc: List[str] = []
    parts = rel_path.split("/")

    # Only allow 4-level paths
    # hair / <gender> / <style> / <color>.png
    if len(parts) != 4:
        return []  # skip preview files

    _, gender, style, filename = parts

    style_label = _normalize_hair_style(style)
    color = _extract_hair_color(filename)

    if color:
        desc.append(f"{color} {style_label}")
    else:
        desc.append(style_label)

    return desc


# -------------------- torso --------------------
def describe_torso(rel_path: str) -> List[str]:
    """Describe torso accessories with accurate material and color."""
    desc: List[str] = []
    path = rel_path.lower()
    filename = rel_path.split("/")[-1]

    # ------------- base type detection -------------
    # capes (normal / tattered / trimmed)
    if "cape" in path:
        if "tatter" in filename:
            base_type = "tattered cape"
        elif "trimcape" in filename or "trimmed" in path:
            base_type = "trimmed cape"
        else:
            base_type = "cape"

    # wings
    elif "wings" in path:
        desc.append("purple wings")
        return desc

    # chain: tabard / mail / clothing
    elif "chain" in path:
        if "tabard" in path:
            base_type = "tabard jacket"
        elif "mail" in filename:
            base_type = "mail"
        else:
            base_type = "clothing"

    # gold / plate / leather armor
    elif "gold" in path or "plate" in path or "leather" in path:
        if "shoulder" in filename:
            base_type = "shoulder armor"
        elif "arms" in filename:
            base_type = "arm armor"
        elif "chest" in filename:
            base_type = "chest armor"
        else:
            base_type = "torso armor"

    # robes
    elif "robe" in path:
        base_type = "robe"

    # dresses
    elif "dress_female" in path:
        if "vest" in filename:
            base_type = "vest"
        elif "dress_w_sash" in filename:
            base_type = "sash dress"
        elif "overskirt" in filename:
            base_type = "overskirt"
        elif "underdress" in filename:
            base_type = "underdress"
        else:
            base_type = "dress"

    # shirts
    elif "shirts" in path:
        if "longsleeve" in path:
            base_type = "long-sleeve shirt"
        elif "sleeveless" in path:
            base_type = "pirate shirt" if "pirate" in filename else "sleeveless shirt"
        else:
            base_type = "shirt"

    # tunics
    elif "tunics" in path:
        base_type = "tunic"

    # default
    else:
        base_type = "clothing"

    # Apply material→color→fallback logic and append the final phrase
    desc.extend(_apply_material_color(base_type, filename, path))

    return desc


# -------------------- legs --------------------
def describe_legs(rel_path: str) -> List[str]:
    """
    Describe leg accessories with accurate material and color:
    - armor
    - pants
    - skirt
    """
    desc: List[str] = []
    path = rel_path.lower()
    filename = rel_path.split("/")[-1]

    # ----- base type detection -----
    if "greaves" in filename:  
        base_type = "greaves"
    elif "metal_pants" in filename:
        base_type = "pants"
    elif "armor" in path:
        base_type = "pants"
    elif "pants" in path:
        base_type = "pants"
    elif "robe_skirt" in filename:
        base_type = "robe skirt"
    elif "skirt" in path:
        base_type = "skirt"
    else:
        base_type = "leg clothing"

    # Apply material→color→fallback logic and append the final phrase
    desc.extend(_apply_material_color(base_type, filename, path))

    return desc


# -------------------- headgear --------------------
def describe_headgear(rel_path: str) -> List[str]:
    """
    Describe headgear type with accurate material and color:
    - bandanas
    - caps
    - helms
    - hoods
    - tiaras
    """
    desc: List[str] = []
    path = rel_path.lower()
    filename = rel_path.split("/")[-1]
    base_type: str

    # -------- base type detection --------
    if "bandana" in path:
        base_type = "bandana"
    elif "caps" in path:
        base_type = "cap"
    elif "helm" in path:
        base_type = "helm"
    elif "hood" in path:
        base_type = "hood"
    elif "tiara" in path:
        base_type = "tiara"
    else:
        base_type = "headgear"

    # Apply material→color→fallback logic and append the final phrase
    desc.extend(_apply_material_color(base_type, filename, path))

    return desc


# -------------------- weapons --------------------
def describe_weapon(rel_path: str) -> List[str]:
    """
    Describe weapon type (simple and material-light).
    """
    desc: List[str] = []
    filename = rel_path.lower().split("/")[-1]
    s = filename

    # ---- spear ----
    if "spear" in s:
        desc.append("spear")

    # ---- dagger ----
    elif "dagger" in s:
        desc.append("dagger")

    # ---- bows ----
    elif "greatbow" in s:
        desc.append("greatbow")
    elif "recurvebow" in s:
        desc.append("recurve bow")
    elif "bow" in s:
        desc.append("bow")

    # ---- wand (with material distinction) ----
    elif "wand" in s:
        material = _extract_material_from_name(s)
        if material:
            desc.append(f"{material} wand")
        else:
            desc.append("wand")

    # ---- shields ----
    elif "shield" in s:
        desc.append("shield")

    # ---- arrows ----
    elif "arrow" in s:
        desc.append("arrows")

    # ---- default ----
    else:
        desc.append("weapon")

    return desc


# -------------------- hands --------------------
def describe_hands(rel_path: str) -> List[str]:
    """
    Describe hand accessories with accurate material and color:
    """
    desc: List[str] = []
    path = rel_path.lower()
    filename = rel_path.split("/")[-1]

    # ---- base type detection ----
    if "bracelet" in path or "bracelet" in filename:
        base_type = "bracelets"
    elif "bandages" in filename:
        base_type = "bandages"
    elif "bracers" in path or "bracers" in filename:
        base_type = "bracers"
    elif "gloves" in path or "gloves" in filename:
        base_type = "gloves"
    else:
        base_type = "hand accessory"

    # -------- material detection (priority) --------
    material = (
        _extract_material_from_name(filename)
        or _extract_material_from_name(path)
    )

    # -------- assemble description --------
    if material:
        # special case (bracer can have material and color)
        if material == "cloth":
            color = (
                _extract_color_from_name(filename)
                or _extract_color_from_name(path)
            )
            if color:
                desc.append(f"{color} {material} {base_type}")
            else:
                desc.append(f"{material} {base_type}")
        else:
            # gold gloves / leather bracers / metal gloves
            desc.append(f"{material} {base_type}")

    else:
        # no material：white gloves, brown hand equipment ...
        desc.append(base_type)
        color = (
            _extract_color_from_name(filename)
            or _extract_color_from_name(path)
        )
        if color:
            desc[-1] = f"{color} {desc[-1]}"
    return desc


# -------------------- feet --------------------
def describe_feet(rel_path: str) -> List[str]:
    """
    Describe feet accessories with accurate material and color:
    - boots
    - shoes
    - slippers
    - ghillies
    """
    desc: List[str] = []
    path = rel_path.lower()
    filename = rel_path.split("/")[-1]

    # ---- base type detection ----
    if "boots" in path:
        base_type = "boots"
    elif "shoes" in path:
        base_type = "shoes"
    elif "slippers" in path:
        base_type = "slippers"
    elif "ghillies" in path:
        base_type = "ghillies"
    else:
        base_type = "footwear"

    # Apply material→color→fallback logic and append the final phrase
    desc.extend(_apply_material_color(base_type, filename, path))

    return desc


# ------------------- MAIN caption builder -------------------
def build_caption_from_combo(combo: dict) -> str:
    """
    Build a natural language caption from the selected layer combo.
    This will be used as text input for diffusion / LoRA training.
    """
    phrases: List[str] = []

    # --- core body info ---
    body_path = combo.get("body")
    if body_path:
        phrases.extend(describe_body(body_path))

    # --- clothing & armor ---
    torso_path = combo.get("torso")
    if torso_path:
        phrases.extend(describe_torso(torso_path))

    legs_path = combo.get("legs")
    if legs_path:
        phrases.extend(describe_legs(legs_path))

    # --- hair & headgear ---
    hair_path = combo.get("hair")
    if hair_path:
        phrases.extend(describe_hair(hair_path))

    headgear_path = combo.get("headgear")
    if headgear_path:
        phrases.extend(describe_headgear(headgear_path))

    # --- hands / feet ---
    hands_path = combo.get("hands")
    if hands_path:
        phrases.extend(describe_hands(hands_path))

    feet_path = combo.get("feet")
    if feet_path:
        phrases.extend(describe_feet(feet_path))

    # --- weapon (optional) ---
    weapon_path = combo.get("weapons")
    if weapon_path:
        phrases.extend(describe_weapon(weapon_path))

    # Deduplicate simple phrases
    phrases = list(dict.fromkeys(phrases))  # preserve order

    # Core description part
    core_desc = ", ".join(phrases) if phrases else "fantasy rpg character"

    # Final caption template
    caption = (
        f"a pixel art fantasy rpg character, {core_desc}, "
        f"four views (front, back, left, right), "
        f"2x2 character sprite sheet, small pixel art, hard edges, no anti-aliasing"
    )

    return caption


def main():
    DIR_ROOT = ASSET_ROOT / DIR_NAME

    print(f"Testing {DIR_NAME} assets under:", DIR_ROOT)

    if not DIR_ROOT.exists():
        print(f"Error: {DIR_NAME} folder not found.")
        return

    print(f"\n=== Testing {DIR_NAME} parsing ===\n")

    # Traverse all PNG files under the selected directory
    for p in DIR_ROOT.rglob("*.png"):
        # Convert to relative path: e.g., feet/shoes/male/brown_shoes_male.png
        rel = p.relative_to(ASSET_ROOT).as_posix()
        
        # Dispatch to the corresponding describe function
        if DIR_NAME == "hair":
            desc = describe_hair(rel)
        elif DIR_NAME == "torso":
            desc = describe_torso(rel)
        elif DIR_NAME == "legs":
            desc = describe_legs(rel)
        elif DIR_NAME == "head":
            desc = describe_headgear(rel)
        elif DIR_NAME == "weapons":
            desc = describe_weapon(rel)
        elif DIR_NAME == "hands":
            desc = describe_hands(rel)
        elif DIR_NAME == "feet":
            desc = describe_feet(rel)
        else:
            desc = ["unknown"]

        print(f"{rel:<90} → {desc}")

    print(f"\n=== {DIR_NAME} test complete ===")


if __name__ == "__main__":
    main()