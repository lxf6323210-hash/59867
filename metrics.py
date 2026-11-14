
import csv
import os

CSV_FILE = os.path.join("results", "experiment_results_200.csv")
OUTPUT_FILE = os.path.join("results", "metrics_summary_200.csv")
SIMILARITY_THRESHOLD = 0.50

LABEL_CORRECT = "CORRECT"
LABEL_ERROR = "ERROR"
LABEL_REVIEW = "REVIEW"


def load_results():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"Cannot find experiment results: {CSV_FILE}")

    with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        raise ValueError("The CSV file contains no experiment records.")

    required = ["Image", "Caption Type", "Similarity", "Qwen Verdict"]
    missing = [name for name in required if name not in (fieldnames or [])]

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    valid_rows = []

    for row in rows:
        try:
            row["_Similarity"] = float(row["Similarity"])
        except (TypeError, ValueError):
            continue

        valid_rows.append(row)

    if not valid_rows:
        raise ValueError("No records with valid similarity scores were found.")

    print("Total CSV records:", len(rows))
    print("Records used:", len(valid_rows))
    print("Invalid records excluded:", len(rows) - len(valid_rows))

    return valid_rows


def get_sbert_prediction(row):
    return LABEL_ERROR if row["_Similarity"] < SIMILARITY_THRESHOLD else LABEL_CORRECT


def get_qwen_prediction(row):
    verdict = row["Qwen Verdict"].strip().upper()

    if verdict == "NOT_SUPPORTED":
        return LABEL_ERROR

    if verdict == "SUPPORTED":
        return LABEL_CORRECT

    return LABEL_REVIEW


def get_combined_prediction(row):
    qwen = row["Qwen Verdict"].strip().upper()
    similarity = row["_Similarity"]

    if qwen == "NOT_SUPPORTED":
        return LABEL_ERROR

    if qwen == "SUPPORTED" and similarity >= SIMILARITY_THRESHOLD:
        return LABEL_CORRECT

    return LABEL_REVIEW


def summarize_predictions(rows, prediction_function):
    counts = {
        LABEL_CORRECT: 0,
        LABEL_ERROR: 0,
        LABEL_REVIEW: 0
    }

    for row in rows:
        counts[prediction_function(row)] += 1

    total = len(rows)
    decided = counts[LABEL_CORRECT] + counts[LABEL_ERROR]

    return {
        "Total": total,
        "Predicted Correct": counts[LABEL_CORRECT],
        "Predicted Error": counts[LABEL_ERROR],
        "Review": counts[LABEL_REVIEW],
        "Decision Coverage": decided / total if total else 0.0
    }


def compare_models(rows):
    agreement = 0
    comparable = 0

    for row in rows:
        sbert_result = get_sbert_prediction(row)
        qwen_result = get_qwen_prediction(row)

        if qwen_result != LABEL_REVIEW:
            comparable += 1

            if sbert_result == qwen_result:
                agreement += 1

    return {
        "Comparable Records": comparable,
        "Agreement": agreement,
        "Disagreement": comparable - agreement,
        "Agreement Rate": agreement / comparable if comparable else 0.0
    }


def print_summary(name, values):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    for key, value in values.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def save_summary(all_summaries, comparison):
    os.makedirs("results", exist_ok=True)

    fields = [
        "Analysis",
        "Total",
        "Predicted Correct",
        "Predicted Error",
        "Review",
        "Decision Coverage",
        "Comparable Records",
        "Agreement",
        "Disagreement",
        "Agreement Rate"
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for name, values in all_summaries.items():
            row = {"Analysis": name}
            row.update(values)
            writer.writerow(row)

        comparison_row = {"Analysis": "SBERT-Qwen Agreement"}
        comparison_row.update(comparison)
        writer.writerow(comparison_row)

    print("\nSummary saved to:", OUTPUT_FILE)


def main():
    print("=" * 60)
    print("CAPTION ERROR DETECTION - PREDICTION ANALYSIS")
    print("=" * 60)

    print("SBERT threshold:", SIMILARITY_THRESHOLD)
    print("Ground Truth annotation: Not required")
    print("Performance accuracy metrics: Not calculated")

    rows = load_results()

    all_summaries = {
        "Sentence-BERT": summarize_predictions(
            rows, get_sbert_prediction
        ),
        "Qwen VLM": summarize_predictions(
            rows, get_qwen_prediction
        ),
        "Combined System": summarize_predictions(
            rows, get_combined_prediction
        )
    }

    for name, values in all_summaries.items():
        print_summary(name, values)

    comparison = compare_models(rows)

    print_summary("SBERT-Qwen Agreement", comparison)

    save_summary(all_summaries, comparison)


if __name__ == "__main__":
    main()
