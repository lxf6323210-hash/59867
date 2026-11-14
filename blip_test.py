
import requests
from PIL import Image
from io import BytesIO

from caption_generator import (
    load_blip,
    generate_caption
)

processor, model, device = load_blip()

image_url = (
    "https://images.unsplash.com/"
    "photo-1518791841217-8f162f1e1131"
)

response = requests.get(
    image_url,
    timeout=30
)

response.raise_for_status()

image = Image.open(
    BytesIO(response.content)
).convert("RGB")

# Generate caption
caption = generate_caption(
    image,
    processor,
    model,
    device
)

print("\n========== BLIP TEST ==========")

print("Device:", device)
print("Generated Caption:", caption)

print("===============================")
