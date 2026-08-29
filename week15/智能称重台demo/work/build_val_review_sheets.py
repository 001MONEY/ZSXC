from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "classification_dataset_from_videos" / "pipeline_manifest_val.csv"
OUTPUT = ROOT / "work" / "classification_val_review"
GROUPS = ("BAG", "BOTTLE", "BOX", "CYLINDER")


def evenly_spaced(rows: list[dict[str, str]], count: int = 5) -> list[dict[str, str]]:
    if len(rows) <= count:
        return rows
    return [rows[round(index * (len(rows) - 1) / (count - 1))] for index in range(count)]


def main() -> None:
    by_sku: dict[str, list[dict[str, str]]] = defaultdict(list)
    with MANIFEST.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["status"] == "saved" and Path(row["output"]).is_file():
                by_sku[row["sku"]].append(row)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    tile_w, tile_h, label_h = 280, 210, 38
    for group in GROUPS:
        skus = sorted(sku for sku in by_sku if sku.startswith(f"{group}_"))
        sheet = Image.new("RGB", (tile_w * 5, (tile_h + label_h) * len(skus)), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, sku in enumerate(skus):
            for column_index, row in enumerate(evenly_spaced(by_sku[sku])):
                with Image.open(row["output"]) as source:
                    image = source.convert("RGB")
                    image.thumbnail((tile_w - 8, tile_h - 8))
                    x = column_index * tile_w + (tile_w - image.width) // 2
                    y0 = row_index * (tile_h + label_h)
                    y = y0 + (tile_h - image.height) // 2
                    sheet.paste(image, (x, y))
                label = f"{sku}  conf={float(row['confidence']):.2f}  {row['selection_method']}"
                draw.text((column_index * tile_w + 4, y0 + tile_h + 4), label, fill="black", font=font)
        sheet.save(OUTPUT / f"{group.lower()}_val_contact_sheet.jpg", quality=92)


if __name__ == "__main__":
    main()
