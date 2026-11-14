
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CSV_FILE = os.path.join("results", "experiment_results_200.csv")
OUTPUT_FOLDER = os.path.join("results", "graphs")
OUTPUT_CSV = os.path.join("results", "model_prediction_cross_tabulation.csv")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(CSV_FILE)

df["Similarity"] = pd.to_numeric(df["Similarity"], errors="coerce")
df = df.dropna(subset=["Similarity"])

df["SBERT Prediction"] = df["Similarity"].apply(
    lambda value: "ERROR" if value < 0.50 else "CORRECT"
)

df["Qwen Prediction"] = df["Qwen Verdict"].str.upper().map({
    "SUPPORTED": "CORRECT",
    "NOT_SUPPORTED": "ERROR"
})

df = df.dropna(subset=["Qwen Prediction"])

labels = ["CORRECT", "ERROR"]

matrix = confusion_matrix(
    df["Qwen Prediction"],
    df["SBERT Prediction"],
    labels=labels
)

result = pd.DataFrame(
    matrix,
    index=["Qwen CORRECT", "Qwen ERROR"],
    columns=["SBERT CORRECT", "SBERT ERROR"]
)

print("=" * 60)
print("SBERT VS QWEN - PREDICTION CROSS-TABULATION")
print("=" * 60)
print(result)
print("\nTotal comparable records:", len(df))

agreement = (df["SBERT Prediction"] == df["Qwen Prediction"]).sum()
disagreement = len(df) - agreement
agreement_rate = agreement / len(df) if len(df) else 0

print("Agreement:", agreement)
print("Disagreement:", disagreement)
print(f"Agreement Rate: {agreement_rate:.4f}")

result.to_csv(OUTPUT_CSV)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels
)

display.plot(cmap="Blues", values_format="d")

plt.title("SBERT vs Qwen - Prediction Cross-Tabulation")
plt.xlabel("Sentence-BERT Prediction")
plt.ylabel("Qwen VLM Prediction")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "model_prediction_cross_tabulation.png"),
    dpi=300
)

plt.show()
plt.close()

print("\nCross-tabulation saved to:", OUTPUT_CSV)
print("Chart saved to:", OUTPUT_FOLDER)
