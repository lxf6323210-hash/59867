
from pathlib import Path
import urllib.request
import zipfile
import tarfile
import random
import shutil
import csv
import io
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parent

DOWNLOAD_DIR = ROOT / "download"
EXTRACT_DIR = ROOT / "extracted"
OUTPUT_DIR = ROOT / "images"

ARCHIVE = DOWNLOAD_DIR / "caltech-101.zip"

URL = (
    "https://data.caltech.edu/records/mzrjq-6wc02/"
    "files/caltech-101.zip?download=1"
)

SEED = 2026

CATEGORY_ALLOCATION = {
    "airplanes": 20,
    "car_side": 20,
    "chair": 20,
    "camera": 20,
    "cup": 20,
    "butterfly": 20,
    "dolphin": 20,
    "elephant": 20,
    "flamingo": 20,
    "motorbikes": 20,
}


def download():

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE.exists():

        if zipfile.is_zipfile(ARCHIVE):
            print("Using existing ZIP archive:")
            print(ARCHIVE)
            return

        print("Existing archive is invalid. Removing it...")
        ARCHIVE.unlink()

    print("\nDownloading official Caltech-101 dataset...")
    print("Download size: approximately 137 MB")
    print("Please wait. Do not close the terminal.\n")

    temp_archive = DOWNLOAD_DIR / "caltech-101.tmp"

    if temp_archive.exists():
        temp_archive.unlink()

    request = urllib.request.Request(
        URL,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            content_type = response.headers.get(
                "Content-Type", ""
            )

            total_size = response.headers.get(
                "Content-Length"
            )

            if total_size:
                total_size = int(total_size)
            else:
                total_size = 0

            downloaded = 0

            with open(temp_archive, "wb") as output:

                while True:

                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    output.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = downloaded / total_size * 100

                        print(
                            f"\rDownloaded: {percent:.1f}%",
                            end=""
                        )

        print()

        if temp_archive.stat().st_size < 1_000_000:
            raise RuntimeError(
                "Downloaded file is too small. "
                "The server may have returned an error page."
            )

        if not zipfile.is_zipfile(temp_archive):
            raise RuntimeError(
                "Downloaded file is not a valid ZIP archive.\n"
                f"Content-Type: {content_type}\n"
                "Please check your internet connection "
                "or download the archive manually."
            )

        temp_archive.replace(ARCHIVE)

        print("\nDownload completed successfully!")

    except Exception:

        if temp_archive.exists():
            temp_archive.unlink()

        raise


def extract_archive(archive_path, destination):

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    if zipfile.is_zipfile(archive_path):

        print("Detected ZIP archive.")
        print("Extracting ZIP...")

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(destination)

    elif tarfile.is_tarfile(archive_path):

        print("Detected TAR archive.")
        print("Extracting TAR...")

        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(destination)

    else:

        raise RuntimeError(
            "Unsupported archive format. "
            "The downloaded file is neither ZIP nor TAR."
        )


def extract():

    marker = EXTRACT_DIR / ".extracted"

    if marker.exists():
        print("\nDataset was already extracted.")
        return

    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)

    EXTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nExtracting main archive...")

    extract_archive(
        ARCHIVE,
        EXTRACT_DIR
    )

    nested_archives = []

    for archive in EXTRACT_DIR.rglob("*"):

        if not archive.is_file():
            continue

        if archive.name.startswith("._"):
            continue

        if "__MACOSX" in archive.parts:
            continue

        if archive.name.lower().endswith((".tar.gz", ".tgz")):

            if tarfile.is_tarfile(archive):
                nested_archives.append(archive)

    for archive in nested_archives:

        if archive.name.lower().endswith(".tar.gz"):
            folder_name = archive.name[:-7]
        else:
            folder_name = archive.name[:-4]

        target = archive.parent / folder_name

        print(
            "\nExtracting nested archive:",
            archive.name
        )

        target.mkdir(
            parents=True,
            exist_ok=True
        )

        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(target)

        print("Extracted:", target)

    marker.write_text(
        "ok",
        encoding="utf-8"
    )

    print("\nExtraction completed!")


def find_category_dir(category):

    matches = []

    for folder in EXTRACT_DIR.rglob("*"):

        if (
            folder.is_dir()
            and folder.name.lower() == category.lower()
        ):

            matches.append(folder)

    if not matches:

        print(
            f"\nERROR: Cannot find category: {category}"
        )

        print("\nAvailable directories:")

        directories = [
            p for p in EXTRACT_DIR.rglob("*")
            if p.is_dir()
        ]

        for folder in directories[:80]:
            print(" -", folder.relative_to(EXTRACT_DIR))

        raise FileNotFoundError(
            f"Category folder not found: {category}"
        )

    return matches[0]


def get_valid_images(folder):

    candidates = []

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    for path in folder.rglob("*"):

        if path.suffix.lower() not in extensions:
            continue

        if path.name.startswith("._") or "__MACOSX" in path.parts:
            continue

        try:

            with Image.open(path) as image:
                image.verify()

            candidates.append(path)

        except Exception:

            print(
                "Skipping invalid image:",
                path.name
            )

    return sorted(candidates)


def generate_dataset():

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rng = random.Random(SEED)

    records = []

    print("\nSelecting images...")
    print("=" * 50)

    for category, count in CATEGORY_ALLOCATION.items():

        print(f"\nCategory: {category}")

        folder = find_category_dir(category)

        candidates = get_valid_images(folder)

        print(
            "Available valid images:",
            len(candidates)
        )

        if len(candidates) < count:

            raise RuntimeError(
                f"{category}: only {len(candidates)} "
                f"valid images found. Need {count}."
            )

        selected = rng.sample(
            candidates,
            count
        )

        for index, source in enumerate(selected, 1):

            filename = f"{category}_{index:02d}.jpg"

            destination = OUTPUT_DIR / filename

            with Image.open(source) as image:

                image.convert("RGB").save(
                    destination,
                    format="JPEG",
                    quality=95
                )

            records.append({
                "filename": filename,
                "category": category,
                "source_file": source.name,
                "ground_truth_caption": "",
                "notes": (
                    "Category label only. "
                    "Generate caption using BLIP. "
                    "Manually verify injected caption errors."
                )
            })

            print(
                f"Saved: {filename}"
            )

    metadata_path = OUTPUT_DIR / "dataset_metadata.csv"

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 50)
    print("DATASET READY!")
    print("=" * 50)

    image_count = sum(
        1 for path in OUTPUT_DIR.iterdir()
        if path.suffix.lower() == ".jpg"
    )

    print("Total images:", image_count)
    print("Total categories:", len(CATEGORY_ALLOCATION))
    print("Images folder:", OUTPUT_DIR)
    print("Metadata file:", metadata_path)

    print("\nCategory summary:")

    for category, count in CATEGORY_ALLOCATION.items():
        print(f"{category}: {count}")

    print("\nNext step:")
    print(
        "Generate captions and run the 200-image experiment."
    )


def main():

    try:

        print("=" * 55)
        print("CALTECH-101 DATASET PREPARATION")
        print("=" * 55)

        download()

        extract()

        generate_dataset()

    except Exception as error:

        print("\nDATASET PREPARATION FAILED!")
        print("=" * 55)
        print("Error:", error)
        print("=" * 55)

        sys.exit(1)


if __name__ == "__main__":
    main()
