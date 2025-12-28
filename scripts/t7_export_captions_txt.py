"""
t7_export_captions_txt.py

Export per-image caption .txt files for BOTH:
  - original captions      (captions.csv)
  - optimized captions     (captions_optimized.csv)

Everything stays under:
    pixel_character_dataset/processed/dataset_4view/

Structure after running:

    dataset_4view/
        images/
        captions.csv
        captions_optimized.csv
        captions_original_txt/
            char_00000.txt   # from captions.csv
            ...
        captions_optimized_txt/
            char_00000.txt   # from captions_optimized.csv
            ...

No images are copied; only write .txt files.
"""

from pathlib import Path
import csv

# ===================== PATH CONFIG =====================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # PIXEL-T2I/

DATASET_ROOT = ROOT / "pixel_character_dataset"
PROCESSED_4VIEW = DATASET_ROOT / "processed" / "dataset_4view"

IMAGE_ROOT = PROCESSED_4VIEW / "images"
CSV_ORIG = PROCESSED_4VIEW / "captions.csv"
CSV_OPT  = PROCESSED_4VIEW / "captions_optimized.csv"

OUT_ORIG_TXT = PROCESSED_4VIEW / "captions_original_txt"
OUT_OPT_TXT  = PROCESSED_4VIEW / "captions_optimized_txt"

print("Dataset root          :", PROCESSED_4VIEW)
print("Images directory      :", IMAGE_ROOT)
print("Original captions CSV :", CSV_ORIG)
print("Optimized captions CSV:", CSV_OPT)
print("Original txt out dir  :", OUT_ORIG_TXT)
print("Optimized txt out dir :", OUT_OPT_TXT)


def export_captions(csv_path: Path, out_dir: Path) -> int:
    """
    Read image_path,text from csv_path and create .txt files in out_dir.

    Does NOT touch/copy images; only writes caption txt files:
        out_dir / <image_stem>.txt
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_ROOT}")

    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "image_path" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("CSV must contain 'image_path' and 'text' columns.")

        for row in reader:
            img_name = Path(row["image_path"]).name
            caption = row["text"]

            # Just write txt: <image_stem>.txt
            txt_path = out_dir / f"{Path(img_name).stem}.txt"
            with txt_path.open("w", encoding="utf-8") as ftxt:
                ftxt.write(caption.strip())

            count += 1

    return count


def main() -> None:
    print("\n=== Exporting ORIGINAL caption txt files ===")
    c1 = export_captions(CSV_ORIG, OUT_ORIG_TXT)
    print(f"Original captions: {c1} txt files written to {OUT_ORIG_TXT}")

    print("\n=== Exporting OPTIMIZED caption txt files ===")
    c2 = export_captions(CSV_OPT, OUT_OPT_TXT)
    print(f"Optimized captions: {c2} txt files written to {OUT_OPT_TXT}")

    print("\nDone. Images remain in:", IMAGE_ROOT)


if __name__ == "__main__":
    main()
