import csv
import os
from datetime import datetime

RESULTS_DIR = "results"
CSV_FILE = os.path.join(RESULTS_DIR, "experiment_results_200.csv")

FIELDNAMES = [
    "Timestamp",
    "Image",
    "Category",
    "Caption Type",
    "Caption",
    "Similarity",
    "Qwen Verdict",
    "Qwen Evidence",
    "Final Decision",
    "Ground Truth",
    "Changed Word",
    "Replacement Word",
    "Fault Injected"
]


def initialize_results(overwrite=False):
    """Create a clean CSV for this 200-image experiment."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if os.path.exists(CSV_FILE) and not overwrite:
        return

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()

    print("Initialized experiment file:", CSV_FILE)


def save_experiment(image_path, result, ground_truth="PENDING",
                    fault_result=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(CSV_FILE):
        initialize_results(overwrite=True)

    filename = os.path.basename(image_path)
    category = filename.rsplit("_", 1)[0] if "_" in filename else "UNKNOWN"
    fault_result = fault_result or {}

    row = {
        "Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Image": image_path,
        "Category": category,
        "Caption Type": result.get("caption_type", ""),
        "Caption": result.get("caption", ""),
        "Similarity": round(float(result.get("similarity", 0.0)), 6),
        "Qwen Verdict": result.get("verdict", "UNCERTAIN"),
        "Qwen Evidence": result.get("evidence", ""),
        "Final Decision": result.get("decision", "NEEDS REVIEW"),
        "Ground Truth": ground_truth,
        "Changed Word": fault_result.get("changed_word") or "",
        "Replacement Word": fault_result.get("replacement_word") or "",
        "Fault Injected": fault_result.get("fault_injected", False)
    }

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writerow(row)

    print("Experiment result saved:", CSV_FILE)
