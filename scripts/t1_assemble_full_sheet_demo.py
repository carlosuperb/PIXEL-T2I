"""
t1_assemble_full_sheet_demo.py

This script demonstrates how to assemble a full LPC-style character sprite sheet
by stacking multiple asset layers (body, legs, torso, hair, etc.) in the correct
order. It is mainly used for debugging: verifying overlay order, checking that
all assets align correctly, and ensuring consistent sprite sheet dimensions
before generating large datasets.
"""

from PIL import Image
import os

# Automatically locate project root based on this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # PIXEL-T2I/scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                     # PIXEL-T2I

# Build BASE and OUT_DIR paths relative to the project root
BASE = os.path.join(
    PROJECT_ROOT,
    "pixel_character_dataset",
    "raw_assets",
    "Universal-LPC-spritesheet"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "pixel_character_dataset",
    "processed",
    "demo"
)
os.makedirs(OUT_DIR, exist_ok=True)

print("BASE   =", BASE)
print("OUTDIR =", OUT_DIR)


def load(path):
    """Load a PNG file and convert it to RGBA."""
    img = Image.open(path).convert("RGBA")
    return img


def create_demo(output_name, body_path, hair_path, torso_path, legs_path, weapon_path):
    """
    Assemble one character sheet and save to OUT_DIR/output_name.png
    """
    layers = [
        load(body_path),
        load(legs_path),
        load(torso_path),
        load(hair_path),
        load(weapon_path),
    ]

    # LPC assets should share identical sheet dimensions
    w, h = layers[0].size
    canvas = Image.new("RGBA", (w, h))

    # Overlay all layers sequentially
    for layer in layers:
        if layer.size != (w, h):
            raise ValueError(f"Layer size mismatch: {layer.size} != {(w, h)}")
        canvas.alpha_composite(layer)

    # Save the assembled sprite sheet
    out_path = os.path.join(OUT_DIR, f"{output_name}.png")
    canvas.save(out_path)
    print("Saved:", out_path)

    # Display the final image size for debugging
    print("Sheet size:", (w, h))


def main():
    # --- Demo 1 ---
    create_demo(
        "character_full_sheet_demo1",
        os.path.join(BASE, "body", "male", "light.png"),
        os.path.join(BASE, "hair", "male", "bangs.png"),
        os.path.join(BASE, "torso", "chain", "mail_male.png"),
        os.path.join(BASE, "legs", "pants", "male", "teal_pants_male.png"),
        os.path.join(BASE, "weapons", "right hand", "male", "spear_male.png"),
    )

    # --- Demo 2 ---
    create_demo(
        "character_full_sheet_demo2",
        os.path.join(BASE, "body", "male", "skeleton.png"),
        os.path.join(BASE, "hair", "male", "long.png"),
        os.path.join(BASE, "torso", "leather", "shoulders_male.png"),
        os.path.join(BASE, "legs", "armor", "male", "metal_pants_male.png"),
        os.path.join(BASE, "weapons", "right hand", "male", "dagger_male.png"),
    )

if __name__ == "__main__":
    main()
