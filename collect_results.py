
import os
import torch
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

IMAGE_DIR = "images"
RESULT_DIR = "results"
EXPERIMENT_FILE = os.path.join(RESULT_DIR, "experiment_results.csv")
OUTPUT_FILE = os.path.join(RESULT_DIR, "upload_package.xlsx")

SKIPPED_IMAGES = [
    "airplanes_04.jpg",
    "airplanes_08.jpg",
    "camera_06.jpg",
    "camera_08.jpg",
    "camera_09.jpg",
    "car_side_05.jpg",
    "car_side_07.jpg",
    "cup_02.jpg",
    "cup_03.jpg",
    "cup_05.jpg",
    "cup_07.jpg",
    "cup_08.jpg",
    "cup_09.jpg",
    "motorbikes_10.jpg"
]

def generate_caption(image_path, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60
        )

    return processor.decode(output[0], skip_special_tokens=True)

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    model.eval()

    skipped_records = []

    for index, filename in enumerate(SKIPPED_IMAGES, start=1):
        image_path = os.path.join(IMAGE_DIR, filename)

        if not os.path.exists(image_path):
            print(f"[{index}/14] Missing: {filename}")
            skipped_records.append({
                "Image": filename,
                "Original Caption": "",
                "Status": "Image file not found"
            })
            continue

        try:
            caption = generate_caption(
                image_path, processor, model, device
            )

            print(f"[{index}/14] {filename}: {caption}")

            skipped_records.append({
                "Image": filename,
                "Original Caption": caption,
                "Status": "Caption generated"
            })

        except Exception as error:
            print(f"Error processing {filename}: {error}")

            skipped_records.append({
                "Image": filename,
                "Original Caption": "",
                "Status": str(error)
            })

    skipped_df = pd.DataFrame(skipped_records)

    if os.path.exists(EXPERIMENT_FILE):
        experiment_df = pd.read_csv(EXPERIMENT_FILE)
    else:
        experiment_df = pd.DataFrame()

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        experiment_df.to_excel(
            writer,
            sheet_name="Experiment Results",
            index=False
        )

        skipped_df.to_excel(
            writer,
            sheet_name="Skipped Images",
            index=False
        )

    print("\n================================")
    print("EXPORT COMPLETED")
    print("Experiment records:", len(experiment_df))
    print("Skipped images:", len(skipped_df))
    print("Excel file:", OUTPUT_FILE)
    print("================================")

if __name__ == "__main__":
    main()
