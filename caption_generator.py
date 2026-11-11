
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

MODEL_NAME = "Salesforce/blip-image-captioning-base"


def load_blip(device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading BLIP model...")
    print("Device:", device)

    processor = BlipProcessor.from_pretrained(
        MODEL_NAME
    )

    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.eval()

    print("BLIP loaded successfully!")

    return processor, model, device


def generate_caption(
    image,
    processor,
    model,
    device
):

    image = image.convert("RGB")

    inputs = processor(
        image,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=20
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption
