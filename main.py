
import os
import torch
from PIL import Image

from caption_generator import load_blip, generate_caption
from semantic_similarity import load_sbert, compute_similarity
from visual_verifier import load_verifier, verify_caption
from fault_injection import inject_fault
from evaluation import save_experiment


SIMILARITY_THRESHOLD = 0.5

IMAGE_FOLDER = "images"
RESULT_FOLDER = "results"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


# ==========================================
# 1. Final Decision
# ==========================================

def make_final_decision(similarity, verdict):

    if verdict == "NOT_SUPPORTED":
        return "POSSIBLE CAPTION ERROR"

    if verdict == "UNCERTAIN":
        return "NEEDS REVIEW"

    if verdict == "SUPPORTED":

        if similarity >= SIMILARITY_THRESHOLD:
            return "CAPTION SUPPORTED"

        return "NEEDS REVIEW"

    return "NEEDS REVIEW"


# ==========================================
# 2. Analyze One Caption
# ==========================================

def analyze_caption(
    image_path,
    reference_caption,
    caption,
    sbert_model,
    qwen_model,
    qwen_processor
):

    similarity = compute_similarity(
        reference_caption,
        caption,
        sbert_model
    )

    verification = verify_caption(
        image_path,
        caption,
        qwen_model,
        qwen_processor
    )

    verdict = verification["verdict"]

    decision = make_final_decision(
        similarity,
        verdict
    )

    return {
        "caption": caption,
        "similarity": similarity,
        "verdict": verdict,
        "evidence": verification["evidence"],
        "decision": decision
    }


# ==========================================
# 3. Get All Images
# ==========================================

def get_image_files():

    if not os.path.exists(IMAGE_FOLDER):
        raise FileNotFoundError(
            f"Image folder not found: {IMAGE_FOLDER}"
        )

    image_files = []

    for filename in sorted(os.listdir(IMAGE_FOLDER)):

        if filename.lower().endswith(SUPPORTED_EXTENSIONS):

            image_path = os.path.join(
                IMAGE_FOLDER,
                filename
            )

            image_files.append(image_path)

    return image_files


# ==========================================
# 4. Process One Image
# ==========================================

def process_image(
    image_path,
    image_index,
    blip_processor,
    blip_model,
    blip_device,
    sbert_model,
    qwen_model,
    qwen_processor
):

    print("\n" + "=" * 65)
    print(f"PROCESSING IMAGE {image_index}: {image_path}")
    print("=" * 65)

    # Open image
    image = Image.open(image_path).convert("RGB")

    print("Image size:", image.size)

    # Generate original caption
    print("\nGenerating Original Caption...")

    generated_caption = generate_caption(
        image,
        blip_processor,
        blip_model,
        blip_device
    )

    print("Original Caption:", generated_caption)

    # Check caption
    if not generated_caption or not generated_caption.strip():

        print("[SKIPPED] BLIP generated an empty caption.")

        return {
            "success": False,
            "reason": "BLIP generated an empty caption."
        }

    # Fault injection
    print("\nApplying Fault Injection...")

    fault_result = inject_fault(
        generated_caption,
        seed=42 + image_index
    )

    if not fault_result["fault_injected"]:

        reason = fault_result.get(
            "reason",
            "No replaceable word found."
        )

        print("\n[SKIPPED] Fault injection failed.")
        print("Image:", image_path)
        print("Original Caption:", generated_caption)
        print("Reason:", reason)

        return {
            "success": False,
            "reason": reason
        }

    modified_caption = fault_result["modified_caption"]

    print("Changed Word:", fault_result["changed_word"])
    print("Replacement Word:", fault_result["replacement_word"])
    print("Modified Caption:", modified_caption)

    # ======================================
    # TEST 1: ORIGINAL CAPTION
    # ======================================

    print("\n" + "-" * 50)
    print("TEST 1: ORIGINAL CAPTION")
    print("-" * 50)

    original_result = analyze_caption(
        image_path,
        generated_caption,
        generated_caption,
        sbert_model,
        qwen_model,
        qwen_processor
    )

    # ======================================
    # TEST 2: MODIFIED CAPTION
    # ======================================

    print("\n" + "-" * 50)
    print("TEST 2: MODIFIED CAPTION")
    print("-" * 50)

    modified_result = analyze_caption(
        image_path,
        generated_caption,
        modified_caption,
        sbert_model,
        qwen_model,
        qwen_processor
    )

    # ======================================
    # Display and Save Results
    # ======================================

    results = [
        ("ORIGINAL", original_result, "CORRECT"),
        ("MODIFIED", modified_result, "ERROR")
    ]

    print("\n" + "=" * 50)
    print("IMAGE RESULTS")
    print("=" * 50)

    for name, result, ground_truth in results:

        print(f"\n{name} CAPTION")
        print("Caption:", result["caption"])
        print(
            "SBERT Similarity:",
            round(result["similarity"], 4)
        )
        print("Qwen Verdict:", result["verdict"])
        print("Qwen Evidence:", result["evidence"])
        print("Final Decision:", result["decision"])

        result["caption_type"] = name

        save_experiment(
            image_path,
            result,
            ground_truth
        )

        print(
            "Experiment result saved:",
            os.path.join(
                RESULT_FOLDER,
                "experiment_results.csv"
            )
        )

    # Fault injection summary
    print("\nFault Injection Summary")
    print("Original:", generated_caption)
    print("Modified:", modified_caption)

    print(
        "Changed:",
        fault_result["changed_word"],
        "->",
        fault_result["replacement_word"]
    )

    return {
        "success": True,
        "reason": "Successfully processed."
    }


# ==========================================
# 5. Main Program
# ==========================================

def main():

    print("=" * 65)
    print("VISUAL CAPTION ERROR DETECTION SYSTEM")
    print("BATCH IMAGE EVALUATION")
    print("=" * 65)

    # Check GPU
    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is not available!"
        )

    print("\nGPU:", torch.cuda.get_device_name(0))

    # Find images
    image_files = get_image_files()

    if not image_files:

        raise RuntimeError(
            "No images found in the images folder!"
        )

    print("\nTotal Images Found:", len(image_files))

    for path in image_files:
        print(" -", path)

    # Create results folder
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    # ======================================
    # Load BLIP
    # ======================================

    print("\n" + "=" * 65)
    print("LOADING BLIP")
    print("=" * 65)

    blip_processor, blip_model, blip_device = load_blip()

    # ======================================
    # Load Sentence-BERT
    # ======================================

    print("\n" + "=" * 65)
    print("LOADING SENTENCE-BERT")
    print("=" * 65)

    sbert_model = load_sbert()

    # ======================================
    # Load Qwen
    # ======================================

    print("\n" + "=" * 65)
    print("LOADING QWEN VISUAL VERIFIER")
    print("=" * 65)

    qwen_model, qwen_processor = load_verifier()

    # ======================================
    # Process All Images
    # ======================================

    successful_images = 0
    skipped_images = 0

    skipped_details = []

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        try:

            result = process_image(
                image_path,
                index,
                blip_processor,
                blip_model,
                blip_device,
                sbert_model,
                qwen_model,
                qwen_processor
            )

            if result["success"]:

                successful_images += 1

            else:

                skipped_images += 1

                skipped_details.append({
                    "image": image_path,
                    "reason": result["reason"]
                })

        except Exception as error:

            skipped_images += 1

            skipped_details.append({
                "image": image_path,
                "reason": str(error)
            })

            print("\n" + "!" * 65)
            print("ERROR PROCESSING IMAGE")
            print("Image:", image_path)
            print("Error:", repr(error))
            print("!" * 65)

            continue

    # ======================================
    # Final Summary
    # ======================================

    print("\n" + "=" * 65)
    print("BATCH EVALUATION COMPLETED")
    print("=" * 65)

    print("Total Images:", len(image_files))
    print("Successfully Processed:", successful_images)
    print("Skipped Images:", skipped_images)

    print(
        "Experiment Records Generated:",
        successful_images * 2
    )

    print(
        "CSV Location:",
        os.path.join(
            RESULT_FOLDER,
            "experiment_results.csv"
        )
    )

    # ======================================
    # Skipped Image Details
    # ======================================

    print("\n" + "=" * 65)
    print("SKIPPED IMAGE DETAILS")
    print("=" * 65)

    if not skipped_details:

        print("No images were skipped.")

    else:

        for item in skipped_details:

            print("\nImage:", item["image"])
            print("Reason:", item["reason"])


    if len(image_files) > 0:

        success_rate = (
            successful_images / len(image_files)
        ) * 100

        print("\nSuccess Rate:", round(success_rate, 2), "%")

    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
