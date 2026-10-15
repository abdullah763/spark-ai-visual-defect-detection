import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.config import METRICS_DIR


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def find_image_files(data_dir: Path) -> list[Path]:
    """Find supported image files recursively."""

    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def audit_dataset(data_dir: Path) -> dict:
    """Inspect image files and return dataset statistics."""

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir}"
        )

    image_files = find_image_files(data_dir)

    records = []
    corrupt_files = []
    hashes = defaultdict(list)

    for image_path in image_files:
        relative_path = image_path.relative_to(data_dir)

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image_format = image.format or "Unknown"
                mode = image.mode

                image.verify()

            file_hash = calculate_sha256(image_path)
            hashes[file_hash].append(str(relative_path))

            top_level_folder = (
                relative_path.parts[0]
                if len(relative_path.parts) > 1
                else "."
            )

            records.append(
                {
                    "path": str(relative_path),
                    "top_level_folder": top_level_folder,
                    "format": image_format,
                    "mode": mode,
                    "width": width,
                    "height": height,
                    "file_size_bytes": image_path.stat().st_size,
                    "sha256": file_hash,
                }
            )

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            corrupt_files.append(
                {
                    "path": str(relative_path),
                    "error": str(exc),
                }
            )

    duplicate_groups = [
        paths
        for paths in hashes.values()
        if len(paths) > 1
    ]

    format_counts = Counter(
        record["format"]
        for record in records
    )

    mode_counts = Counter(
        record["mode"]
        for record in records
    )

    folder_counts = Counter(
        record["top_level_folder"]
        for record in records
    )

    widths = [
        record["width"]
        for record in records
    ]

    heights = [
        record["height"]
        for record in records
    ]

    summary = {
        "dataset_directory": str(data_dir.resolve()),
        "discovered_image_files": len(image_files),
        "valid_images": len(records),
        "corrupt_images": len(corrupt_files),
        "format_counts": dict(format_counts),
        "mode_counts": dict(mode_counts),
        "top_level_folder_counts": dict(folder_counts),
        "width": {
            "min": min(widths) if widths else None,
            "max": max(widths) if widths else None,
        },
        "height": {
            "min": min(heights) if heights else None,
            "max": max(heights) if heights else None,
        },
        "exact_duplicate_groups": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "corrupt_files": corrupt_files,
    }

    return {
        "summary": summary,
        "records": records,
    }


def save_audit_results(audit_result: dict) -> None:
    """Save dataset audit results as JSON and CSV."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = METRICS_DIR / "dataset_audit_summary.json"
    csv_path = METRICS_DIR / "dataset_audit_images.csv"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit_result["summary"],
            file,
            indent=2,
        )

    records = audit_result["records"]

    if records:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=records[0].keys(),
            )

            writer.writeheader()
            writer.writerows(records)

    print(f"Summary saved to: {summary_path}")
    print(f"Image details saved to: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit an image dataset before model training."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing dataset images.",
    )

    args = parser.parse_args()

    result = audit_dataset(args.data_dir)

    print(
        json.dumps(
            result["summary"],
            indent=2,
        )
    )

    save_audit_results(result)


if __name__ == "__main__":
    main()
