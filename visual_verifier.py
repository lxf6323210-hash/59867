from PIL import Image
import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_verifier():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available!")

    print("Loading Qwen2.5-VL...")
    print("GPU:", torch.cuda.get_device_name(0))

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )

    model.eval()

    print("Qwen2.5-VL loaded successfully!")

    return model, processor


def verify_caption(
    image_path,
    caption,
    model,
    processor
):

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((448, 448))

    print("Image size:", image.size)

    prompt = f"""
You are an independent visual caption verifier.

Examine the provided image carefully.

Determine whether the following caption is visually
supported by the image.

Caption: "{caption}"

Rules:
1. Only use evidence visible in the image.
2. Do not assume objects that cannot be seen.
3. Distinguish unsupported claims from clear contradictions.
4. If the image is ambiguous, use UNCERTAIN.

Return exactly this format:

VERDICT: SUPPORTED or NOT_SUPPORTED or UNCERTAIN
EVIDENCE: One concise sentence explaining your decision.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    torch.cuda.empty_cache()

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

    output_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    verdict = "UNCERTAIN"

    for label in [
        "SUPPORTED",
        "NOT_SUPPORTED",
        "UNCERTAIN"
    ]:

        if f"VERDICT: {label}" in response:
            verdict = label
            break

    return {
        "caption": caption,
        "verdict": verdict,
        "evidence": response,
        "raw_response": response
    }

