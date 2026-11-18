
import os
import csv
from PIL import Image

from caption_generator import load_blip, generate_caption

SKIPPED_IMAGES = [
    "airplanes_04.jpg",
    "airplanes_08.jpg",
    "airplanes_13.jpg",
    "butterfly_05.jpg",
    "camera_06.jpg",
    "camera_08.jpg",
    "camera_09.jpg",
    "camera_10.jpg",
    "camera_14.jpg",
    "camera_16.jpg",
    "car_side_06.jpg",
    "car_side_08.jpg",
    "car_side_19.jpg",
    "chair_19.jpg",
    "cup_01.jpg",
    "cup_03.jpg",
    "cup_04.jpg",
    "cup_07.jpg",
    "cup_09.jpg",
    "cup_17.jpg",
    "cup_19.jpg",
    "motorbikes_03.jpg"
]

def main():
    processor, model, device = load_blip()

    output = []

    for filename in SKIPPED_IMAGES:
        path = os.path.join("images", filename)

        with Image.open(path) as img:
            image = img.convert("RGB")
            caption = generate_caption(
                image, processor, model, device
            )

        print(f"{filename}: {caption}")

        output.append({
            "Image": filename,
            "Original Caption": caption
        })

    with open(
        "results/skipped_captions.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Image", "Original Caption"]
        )
        writer.writeheader()
        writer.writerows(output)

    print("\nSaved to results/skipped_captions.csv")

if __name__ == "__main__":
    main()
