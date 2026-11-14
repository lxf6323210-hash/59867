import os
import torch
from PIL import Image

from caption_generator import load_blip, generate_caption
from semantic_similarity import load_sbert, compute_similarity
from visual_verifier import load_verifier, verify_caption
from fault_injection import inject_fault
from evaluation import save_experiment, initialize_results, CSV_FILE

# Keep this threshold identical to metrics.py.
# It is an initial experimental setting, NOT a calibrated threshold.
SIMILARITY_THRESHOLD = 0.50

IMAGE_FOLDER = "images"
RESULT_FOLDER = "results"
EXPECTED_IMAGE_COUNT = 200
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def make_final_decision(similarity, verdict):
    verdict = str(verdict).strip().upper()

    if verdict == "NOT_SUPPORTED":
        return "POSSIBLE CAPTION ERROR"
    if verdict == "UNCERTAIN":
        return "NEEDS REVIEW"
    if verdict == "SUPPORTED" and similarity >= SIMILARITY_THRESHOLD:
        return "CAPTION SUPPORTED"
    return "NEEDS REVIEW"


def analyze_caption(image_path, reference_caption, caption,
                    sbert_model, qwen_model, qwen_processor):
    similarity = compute_similarity(reference_caption, caption, sbert_model)

    verification = verify_caption(
        image_path, caption, qwen_model, qwen_processor
    )

    verdict = verification.get("verdict", "UNCERTAIN")
    return {
        "caption": caption,
        "similarity": similarity,
        "verdict": verdict,
        "evidence": verification.get("evidence", ""),
        "decision": make_final_decision(similarity, verdict)
    }


def get_image_files():
    if not os.path.isdir(IMAGE_FOLDER):
        raise FileNotFoundError(f"Image folder not found: {IMAGE_FOLDER}")

    files = [
        os.path.join(IMAGE_FOLDER, name)
        for name in sorted(os.listdir(IMAGE_FOLDER))
        if name.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    return files


def process_image(image_path, image_index, blip_processor, blip_model,
                  blip_device, sbert_model, qwen_model, qwen_processor):
    print("\n" + "=" * 65)
    print(f"PROCESSING IMAGE {image_index}: {image_path}")
    print("=" * 65)

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    print("Image size:", image.size)
    generated_caption = generate_caption(
        image, blip_processor, blip_model, blip_device
    )
    print("Original Caption:", generated_caption)

    if not generated_caption or not generated_caption.strip():
        return {"success": False, "reason": "BLIP generated an empty caption."}

    fault_result = inject_fault(generated_caption, seed=42 + image_index)

    if not fault_result["fault_injected"]:
        reason = fault_result.get("reason", "No replaceable word found.")
        print("[SKIPPED] Fault injection failed:", reason)
        return {"success": False, "reason": reason}

    modified_caption = fault_result["modified_caption"]
    print("Changed Word:", fault_result["changed_word"])
    print("Replacement Word:", fault_result["replacement_word"])
    print("Modified Caption:", modified_caption)

    original_result = analyze_caption(
        image_path, generated_caption, generated_caption,
        sbert_model, qwen_model, qwen_processor
    )
    modified_result = analyze_caption(
        image_path, generated_caption, modified_caption,
        sbert_model, qwen_model, qwen_processor
    )

    # IMPORTANT:
    # Do not automatically label original as CORRECT or modified as ERROR.
    # Both labels remain PENDING until a human checks the image and caption.
    for caption_type, result in [
        ("ORIGINAL", original_result),
        ("MODIFIED", modified_result)
    ]:
        result["caption_type"] = caption_type

        print(f"\n{caption_type} CAPTION")
        print("Caption:", result["caption"])
        print("SBERT Similarity:", round(result["similarity"], 4))
        print("Qwen Verdict:", result["verdict"])
        print("Qwen Evidence:", result["evidence"])
        print("System Decision:", result["decision"])

        save_experiment(
            image_path=image_path,
            result=result,
            ground_truth="PENDING",
            fault_result=fault_result
        )

    return {"success": True, "reason": "Successfully processed."}


def main():
    print("=" * 65)
    print("VISUAL CAPTION ERROR DETECTION SYSTEM")
    print("200-IMAGE EXPERIMENT")
    print("=" * 65)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available!")
    print("GPU:", torch.cuda.get_device_name(0))

    image_files = get_image_files()
    if not image_files:
        raise RuntimeError("No images found in the images folder!")

    print("\nTotal Images Found:", len(image_files))
    if len(image_files) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGE_COUNT} images, but found "
            f"{len(image_files)}. Check images/ before running."
        )

    # A clean run is created intentionally. This does NOT delete images/.
    # Existing experiment_results_200.csv will be overwritten.
    confirm = input(
        "\nThis will overwrite results/experiment_results_200.csv. "
        "Type YES to continue: "
    ).strip()
    if confirm != "YES":
        print("Cancelled. Existing results were not changed.")
        return

    initialize_results(overwrite=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    print("\nLoading BLIP...")
    blip_processor, blip_model, blip_device = load_blip()

    print("\nLoading Sentence-BERT...")
    sbert_model = load_sbert()

    print("\nLoading Qwen visual verifier...")
    qwen_model, qwen_processor = load_verifier()

    successful = 0
    skipped = []

    for index, image_path in enumerate(image_files, start=1):
        try:
            outcome = process_image(
                image_path, index,
                blip_processor, blip_model, blip_device,
                sbert_model, qwen_model, qwen_processor
            )
            if outcome["success"]:
                successful += 1
            else:
                skipped.append((image_path, outcome["reason"]))
        except Exception as error:
            skipped.append((image_path, repr(error)))
            print("\n[ERROR]", image_path, repr(error))
            continue

    print("\n" + "=" * 65)
    print("BATCH EXPERIMENT COMPLETED")
    print("=" * 65)
    print("Total images:", len(image_files))
    print("Successfully processed:", successful)
    print("Skipped images:", len(skipped))
    print("Expected records from successful images:", successful * 2)
    print("CSV:", CSV_FILE)

    if skipped:
        print("\nSkipped image details:")
        for path, reason in skipped:
            print("-", path, "|", reason)

    print(
        "\nNOTE: Ground Truth is PENDING. Manually annotate the CSV "
        "before running metrics.py."
    )


if __name__ == "__main__":
    main()
