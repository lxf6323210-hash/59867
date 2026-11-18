
import gc
import json
import re
import torch
from PIL import Image
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

CONTRADICTION_PAIRS = [
    (
        ["flying", "in the sky", "airborne"],
        ["parked on the ground", "on the ground", "not flying", "not airborne"]
    ),
    (
        ["running"],
        ["standing still", "not running", "stationary"]
    ),
    (
        ["sitting"],
        ["standing", "not sitting"]
    ),
    (
        ["swimming"],
        ["on dry land", "not swimming", "out of the water"]
    ),
    (
        ["in the water", "underwater"],
        ["on dry land", "not in the water"]
    ),
    (
        ["open"],
        ["closed", "shut"]
    ),
    (
        ["daytime", "during the day"],
        ["at night", "nighttime", "dark outside"]
    ),
    (
        ["indoors", "inside"],
        ["outdoors", "outside"]
    )
]


class LlamaVerifier:

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is not available.")

        print("Loading Llama 3.2 11B Vision...")

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        self.model.eval()

        print("Llama model loaded successfully.")

    def verify(self, image_path, caption):

        image = Image.open(image_path).convert("RGB")
        image.thumbnail((448, 448))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "You are a strict image-caption consistency evaluator.\n"
                            "Inspect the image carefully and compare it with the caption.\n"
                            "SUPPORTED means the main description agrees with visible evidence.\n"
                            "UNSUPPORTED means the caption clearly contradicts the image.\n"
                            "UNCERTAIN means there is insufficient visual evidence.\n"
                            "Do not assume an action is happening because it is common.\n"
                            "Use only visible evidence.\n"
                            "Return exactly one JSON object with these fields:\n"
                            '{"verdict":"SUPPORTED, UNSUPPORTED, or UNCERTAIN",'
                            '"evidence":"Brief visual explanation"}\n'
                            "The verdict and evidence must agree.\n"
                            "Do not include Markdown or text outside JSON.\n"
                            f"Caption: {caption}"
                        )
                    }
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        inputs = self.processor(
            image,
            prompt,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.model.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )

        generated_tokens = output[0][
            inputs["input_ids"].shape[1]:
        ]

        response = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        raw_verdict = "UNCERTAIN"
        final_verdict = "UNCERTAIN"
        evidence = response
        consistency_flag = "VALID"

        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)

            if not match:
                consistency_flag = "INVALID_JSON"
            else:
                parsed = json.loads(match.group(0))

                candidate = str(
                    parsed.get("verdict", "")
                ).strip().upper()

                candidate = re.sub(r"[^A-Z]", "", candidate)

                if candidate in (
                    "SUPPORTED",
                    "UNSUPPORTED",
                    "UNCERTAIN"
                ):
                    raw_verdict = candidate
                    final_verdict = candidate
                else:
                    consistency_flag = "INVALID_VERDICT"

                evidence = str(
                    parsed.get("evidence", response)
                ).strip()

        except (json.JSONDecodeError, TypeError, ValueError):
            consistency_flag = "INVALID_JSON"

        caption_lower = caption.lower()
        evidence_lower = evidence.lower()

        if raw_verdict == "SUPPORTED":

            contradiction_found = any(
                any(term in caption_lower for term in caption_terms)
                and any(term in evidence_lower for term in evidence_terms)
                for caption_terms, evidence_terms in CONTRADICTION_PAIRS
            )

            if contradiction_found:
                final_verdict = "UNCERTAIN"
                consistency_flag = "CONTRADICTION_DETECTED"

        return {
            "raw_verdict": raw_verdict,
            "verdict": final_verdict,
            "evidence": evidence,
            "raw_response": response,
            "consistency": consistency_flag
        }

    def unload(self):

        if hasattr(self, "model"):
            del self.model

        if hasattr(self, "processor"):
            del self.processor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Llama model unloaded.")
