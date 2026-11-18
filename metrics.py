
import os
import pandas as pd
import numpy as np

INPUT_FILE = "results/experiment_results_200_with_llama.csv"
OUTPUT_DIR = "results"
THRESHOLD = 0.50

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "Similarity",
    "Qwen Verdict",
    "Llama Verdict",
    "Llama Status",
    "Llama Consistency"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Similarity"] = pd.to_numeric(df["Similarity"], errors="coerce")

def normalize_qwen(value):
    if pd.isna(value):
        return "UNKNOWN"

    value = str(value).strip().upper()

    if value in ["SUPPORTED", "SUPPORT"]:
        return "SUPPORTED"

    if value in ["NOT_SUPPORTED", "UNSUPPORTED", "NOT SUPPORTED"]:
        return "UNSUPPORTED"

    if value in ["UNCERTAIN", "UNKNOWN"]:
        return "UNCERTAIN"

    return "UNKNOWN"

def normalize_llama(value):
    if pd.isna(value):
        return "UNKNOWN"

    value = str(value).strip().upper()

    if value in ["SUPPORTED", "SUPPORT"]:
        return "SUPPORTED"

    if value in ["UNSUPPORTED", "NOT_SUPPORTED", "NOT SUPPORTED"]:
        return "UNSUPPORTED"

    if value in ["UNCERTAIN", "UNKNOWN"]:
        return "UNCERTAIN"

    return "UNKNOWN"

df["SBERT Prediction"] = np.where(
    df["Similarity"] >= THRESHOLD,
    "HIGH_SIMILARITY",
    "LOW_SIMILARITY"
)

df.loc[df["Similarity"].isna(), "SBERT Prediction"] = "UNKNOWN"

df["Qwen Prediction"] = df["Qwen Verdict"].apply(normalize_qwen)
df["Llama Prediction"] = df["Llama Verdict"].apply(normalize_llama)

sbert_counts = df["SBERT Prediction"].value_counts()
qwen_counts = df["Qwen Prediction"].value_counts()
llama_counts = df["Llama Prediction"].value_counts()

def create_distribution(model_name, counts, labels):
    records = []

    total = len(df)

    for label in labels:
        count = int(counts.get(label, 0))
        percentage = count / total * 100 if total else 0

        records.append({
            "Model": model_name,
            "Prediction": label,
            "Count": count,
            "Percentage": round(percentage, 2)
        })

    return records

distribution = []

distribution += create_distribution(
    "SBERT",
    sbert_counts,
    ["HIGH_SIMILARITY", "LOW_SIMILARITY", "UNKNOWN"]
)

distribution += create_distribution(
    "Qwen",
    qwen_counts,
    ["SUPPORTED", "UNSUPPORTED", "UNCERTAIN", "UNKNOWN"]
)

distribution += create_distribution(
    "Llama",
    llama_counts,
    ["SUPPORTED", "UNSUPPORTED", "UNCERTAIN", "UNKNOWN"]
)

distribution_df = pd.DataFrame(distribution)

distribution_df.to_csv(
    os.path.join(OUTPUT_DIR, "prediction_distribution_3models.csv"),
    index=False
)

visual_labels = ["SUPPORTED", "UNSUPPORTED", "UNCERTAIN"]

valid_visual = df[
    df["Qwen Prediction"].isin(visual_labels)
    & df["Llama Prediction"].isin(visual_labels)
]

agreement = (
    valid_visual["Qwen Prediction"]
    == valid_visual["Llama Prediction"]
)

agreement_count = int(agreement.sum())
disagreement_count = int((~agreement).sum())
comparable_count = len(valid_visual)

agreement_rate = (
    agreement_count / comparable_count * 100
    if comparable_count else 0
)

pairwise_df = pd.DataFrame([{
    "Model A": "Qwen",
    "Model B": "Llama",
    "Comparable Records": comparable_count,
    "Agreement": agreement_count,
    "Disagreement": disagreement_count,
    "Agreement Rate (%)": round(agreement_rate, 2)
}])

pairwise_df.to_csv(
    os.path.join(OUTPUT_DIR, "visual_model_agreement.csv"),
    index=False
)

cross_tab = pd.crosstab(
    valid_visual["Qwen Prediction"],
    valid_visual["Llama Prediction"]
)

cross_tab = cross_tab.reindex(
    index=visual_labels,
    columns=visual_labels,
    fill_value=0
)

cross_tab.to_csv(
    os.path.join(OUTPUT_DIR, "qwen_llama_cross_tabulation.csv")
)

all_three = df[
    df["Qwen Prediction"].isin(visual_labels)
    & df["Llama Prediction"].isin(visual_labels)
    & df["SBERT Prediction"].isin(["HIGH_SIMILARITY", "LOW_SIMILARITY"])
].copy()

three_model_summary = {
    "Total Records": len(df),
    "Comparable Visual Model Records": comparable_count,
    "Qwen-Llama Agreement": agreement_count,
    "Qwen-Llama Disagreement": disagreement_count,
    "Qwen-Llama Agreement Rate (%)": round(agreement_rate, 2),
    "SBERT High Similarity": int((df["SBERT Prediction"] == "HIGH_SIMILARITY").sum()),
    "SBERT Low Similarity": int((df["SBERT Prediction"] == "LOW_SIMILARITY").sum()),
    "Qwen Supported": int((df["Qwen Prediction"] == "SUPPORTED").sum()),
    "Qwen Unsupported": int((df["Qwen Prediction"] == "UNSUPPORTED").sum()),
    "Qwen Uncertain": int((df["Qwen Prediction"] == "UNCERTAIN").sum()),
    "Llama Supported": int((df["Llama Prediction"] == "SUPPORTED").sum()),
    "Llama Unsupported": int((df["Llama Prediction"] == "UNSUPPORTED").sum()),
    "Llama Uncertain": int((df["Llama Prediction"] == "UNCERTAIN").sum()),
    "Llama Invalid JSON": int((df["Llama Consistency"] == "INVALID_JSON").sum()),
    "Llama Contradictions": int((df["Llama Consistency"] == "CONTRADICTION_DETECTED").sum()),
    "Llama Successful Inferences": int((df["Llama Status"] == "SUCCESS").sum())
}

summary_df = pd.DataFrame(
    list(three_model_summary.items()),
    columns=["Metric", "Value"]
)

summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "three_model_summary_corrected.csv"),
    index=False
)

print("\n========== SBERT TEXT SIMILARITY ==========")
print(sbert_counts.to_string())

print("\n========== QWEN VISUAL VERIFICATION ==========")
print(qwen_counts.to_string())

print("\n========== LLAMA VISUAL VERIFICATION ==========")
print(llama_counts.to_string())

print("\n========== QWEN VS LLAMA AGREEMENT ==========")
print(pairwise_df.to_string(index=False))

print("\n========== QWEN VS LLAMA CROSS-TABULATION ==========")
print(cross_tab)

print("\n========== LLAMA QUALITY CHECK ==========")
print("Successful inferences:", three_model_summary["Llama Successful Inferences"])
print("Invalid JSON:", three_model_summary["Llama Invalid JSON"])
print("Contradictions:", three_model_summary["Llama Contradictions"])

print("\nGenerated files:")
print("results/prediction_distribution_3models.csv")
print("results/visual_model_agreement.csv")
print("results/qwen_llama_cross_tabulation.csv")
print("results/three_model_summary_corrected.csv")

print("\nMetrics analysis completed.")
