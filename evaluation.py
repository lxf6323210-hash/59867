
import os
import csv
from datetime import datetime


RESULTS_DIR = "results"
CSV_FILE = os.path.join(
    RESULTS_DIR,
    "experiment_results.csv"
)


def save_experiment(image_path, result, ground_truth):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    file_exists = os.path.isfile(CSV_FILE)

    with open(
        CSV_FILE,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Image",
                "Caption Type",
                "Caption",
                "Similarity",
                "Qwen Verdict",
                "Final Decision",
                "Ground Truth"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            image_path,
            result["caption_type"],
            result["caption"],
            round(result["similarity"], 4),
            result["verdict"],
            result["decision"],
            ground_truth
        ])

    print("Experiment result saved:", CSV_FILE)
