#!/usr/bin/env python3
"""Filter a COCO JSON to person-only and copy the corresponding images."""

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ann",
        default=str(here / "annotations" / "instances_val2017.json"),
        help="Input COCO annotation JSON",
    )
    parser.add_argument(
        "--images-dir",
        default=str(here / "val2017"),
        help="Directory containing the source images",
    )
    parser.add_argument(
        "--out-ann",
        default=str(here / "annotations" / "instances_val2017_person.json"),
        help="Output annotation JSON path",
    )
    parser.add_argument(
        "--out-images-dir",
        default=str(here / "val2017_person"),
        help="Directory to copy filtered images into",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Keep only the first N images (sorted by image id)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.ann) as f:
        coco = json.load(f)

    person_cat_id = next(
        c["id"] for c in coco["categories"] if c["name"] == "person"
    )

    person_anns = [a for a in coco["annotations"] if a["category_id"] == person_cat_id]
    valid_image_ids = {a["image_id"] for a in person_anns}
    filtered_images = sorted(
        [img for img in coco["images"] if img["id"] in valid_image_ids],
        key=lambda x: x["id"],
    )
    if args.max_images is not None:
        filtered_images = filtered_images[: args.max_images]
        keep_ids = {img["id"] for img in filtered_images}
        person_anns = [a for a in person_anns if a["image_id"] in keep_ids]

    out_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": [c for c in coco["categories"] if c["id"] == person_cat_id],
        "images": filtered_images,
        "annotations": person_anns,
    }

    Path(args.out_ann).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_ann, "w") as f:
        json.dump(out_coco, f)
    print(f"Wrote {len(filtered_images)} images, {len(person_anns)} annotations -> {args.out_ann}")

    out_dir = Path(args.out_images_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in filtered_images:
        src = Path(args.images_dir) / img["file_name"]
        dst = out_dir / img["file_name"]
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"Copied {len(filtered_images)} images -> {out_dir}")


if __name__ == "__main__":
    main()
