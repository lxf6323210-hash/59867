
import os
import requests

from visual_verifier import (
    load_verifier,
    verify_caption
)


os.makedirs("images", exist_ok=True)

image_path = "images/test.jpg"

image_url = (
    "https://images.unsplash.com/"
    "photo-1518791841217-8f162f1e1131"
)

if not os.path.exists(image_path):

    print("Downloading test image...")

    response = requests.get(
        image_url,
        timeout=30
    )

    response.raise_for_status()

    with open(image_path, "wb") as f:
        f.write(response.content)


model, processor = load_verifier()

caption = "A person is standing behind the house."

result = verify_caption(
    image_path,
    caption,
    model,
    processor
)

print("\n========== VLM VERIFIER TEST ==========")

print("Caption:", result["caption"])
print("Verdict:", result["verdict"])
print("Evidence:", result["evidence"])

print("=======================================")
