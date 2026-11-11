
import csv
import os


CSV_FILE = os.path.join("results", "experiment_results.csv")
OUTPUT_FILE = os.path.join("results", "metrics_summary.csv")

# Initial experimental threshold.
# This value should be calibrated with more test images.
SIMILARITY_THRESHOLD = 0.95

LABEL_CORRECT = "CORRECT"
LABEL_ERROR = "ERROR"
LABEL_REVIEW = "REVIEW"


# ==========================================
# 1. Load Experiment Results
# ==========================================

def load_results():

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"Cannot find experiment results: {CSV_FILE}"
        )

    with open(
        CSV_FILE,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError(
            "The CSV file contains no experiment records."
        )

    required_columns = [
        "Timestamp",
        "Image",
        "Caption Type",
        "Similarity",
        "Qwen Verdict",
        "Ground Truth"
    ]

    missing = [
        column for column in required_columns
        if column not in rows[0]
    ]

    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}"
        )

    print("Original CSV records:", len(rows))

    # Keep only the latest record for each image and caption type.
    latest_records = {}

    for row in rows:

        key = (
            row["Image"].strip(),
            row["Caption Type"].strip().upper()
        )

        timestamp = row["Timestamp"].strip()

        if (
            key not in latest_records
            or timestamp >= latest_records[key]["Timestamp"].strip()
        ):
            latest_records[key] = row

    rows = list(latest_records.values())

    print("Records after removing old duplicates:", len(rows))

    return rows


# ==========================================
# 2. Sentence-BERT Prediction
# ==========================================

def get_sbert_prediction(row):

    similarity = float(row["Similarity"])

    if similarity < SIMILARITY_THRESHOLD:
        return LABEL_ERROR

    return LABEL_CORRECT


# ==========================================
# 3. Qwen VLM Prediction
# ==========================================

def get_qwen_prediction(row):

    verdict = row["Qwen Verdict"].strip().upper()

    if verdict == "NOT_SUPPORTED":
        return LABEL_ERROR

    if verdict == "SUPPORTED":
        return LABEL_CORRECT

    return LABEL_REVIEW


# ==========================================
# 4. Combined System Prediction
# ==========================================

def get_combined_prediction(row):

    similarity = float(row["Similarity"])

    qwen_verdict = row["Qwen Verdict"].strip().upper()

    # Qwen directly detects an unsupported caption.
    if qwen_verdict == "NOT_SUPPORTED":
        return LABEL_ERROR

    # Both models must support the caption.
    if qwen_verdict == "SUPPORTED":

        if similarity >= SIMILARITY_THRESHOLD:
            return LABEL_CORRECT

        # The two signals disagree.
        return LABEL_REVIEW

    return LABEL_REVIEW


# ==========================================
# 5. Calculate Evaluation Metrics
# ==========================================

def calculate_metrics(rows, prediction_function):

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    review = 0

    for row in rows:

        actual = row["Ground Truth"].strip().upper()

        predicted = prediction_function(row)

        if actual not in (LABEL_CORRECT, LABEL_ERROR):
            continue

        if predicted == LABEL_REVIEW:
            review += 1
            continue

        if actual == LABEL_ERROR and predicted == LABEL_ERROR:
            tp += 1

        elif actual == LABEL_CORRECT and predicted == LABEL_CORRECT:
            tn += 1

        elif actual == LABEL_CORRECT and predicted == LABEL_ERROR:
            fp += 1

        elif actual == LABEL_ERROR and predicted == LABEL_CORRECT:
            fn += 1

    evaluated = tp + tn + fp + fn
    total = evaluated + review

    accuracy = (
        (tp + tn) / evaluated
        if evaluated else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) else 0
    )

    coverage = (
        evaluated / total
        if total else 0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Review": review,
        "Evaluated": evaluated,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "False Positive Rate": false_positive_rate,
        "Coverage": coverage
    }


# ==========================================
# 6. Print Results
# ==========================================

def print_metrics(name, metrics):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("True Positive (TP):", metrics["TP"])
    print("True Negative (TN):", metrics["TN"])
    print("False Positive (FP):", metrics["FP"])
    print("False Negative (FN):", metrics["FN"])
    print("Needs Review:", metrics["Review"])
    print("Evaluated:", metrics["Evaluated"])

    print("\nEvaluation Metrics")

    print(f"Accuracy:             {metrics['Accuracy']:.4f}")
    print(f"Precision:            {metrics['Precision']:.4f}")
    print(f"Recall:               {metrics['Recall']:.4f}")
    print(f"F1-score:             {metrics['F1-score']:.4f}")

    print(
        f"False Positive Rate:  "
        f"{metrics['False Positive Rate']:.4f}"
    )

    print(f"Coverage:             {metrics['Coverage']:.4f}")


# ==========================================
# 7. Save Metrics to CSV
# ==========================================

def save_metrics(all_metrics):

    os.makedirs("results", exist_ok=True)

    fieldnames = [
        "Model",
        "TP",
        "TN",
        "FP",
        "FN",
        "Review",
        "Evaluated",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "False Positive Rate",
        "Coverage"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for model_name, metrics in all_metrics.items():

            row = {"Model": model_name}
            row.update(metrics)

            writer.writerow(row)

    print("\nMetrics saved to:", OUTPUT_FILE)


# ==========================================
# 8. Main
# ==========================================

def main():

    print("=" * 60)
    print("CAPTION ERROR DETECTION - MODEL EVALUATION")
    print("=" * 60)

    rows = load_results()

    print("\nTotal experiment records:", len(rows))
    print("SBERT similarity threshold:", SIMILARITY_THRESHOLD)

    all_metrics = {}

    # Sentence-BERT
    sbert_metrics = calculate_metrics(
        rows,
        get_sbert_prediction
    )

    all_metrics["Sentence-BERT"] = sbert_metrics

    # Qwen VLM
    qwen_metrics = calculate_metrics(
        rows,
        get_qwen_prediction
    )

    all_metrics["Qwen VLM"] = qwen_metrics

    # Combined System
    combined_metrics = calculate_metrics(
        rows,
        get_combined_prediction
    )

    all_metrics["Combined System"] = combined_metrics

    # Display all results
    for name, metrics in all_metrics.items():
        print_metrics(name, metrics)

    # Save results
    save_metrics(all_metrics)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
