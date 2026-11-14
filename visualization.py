
import os
import pandas as pd
import matplotlib.pyplot as plt

METRICS_FILE = os.path.join("results", "metrics_summary_200.csv")
OUTPUT_FOLDER = os.path.join("results", "graphs")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(METRICS_FILE)

model_df = df[df["Analysis"].isin([
    "Sentence-BERT",
    "Qwen VLM",
    "Combined System"
])].copy()

models = model_df["Analysis"].tolist()

prediction_columns = [
    "Predicted Correct",
    "Predicted Error",
    "Review"
]

colors = None

ax = model_df.set_index("Analysis")[prediction_columns].plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title("Caption Error Detection - Prediction Distribution")
ax.set_ylabel("Number of Predictions")
ax.set_xlabel("Model")
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_FOLDER, "prediction_distribution.png"),
    dpi=300
)
plt.show()
plt.close()

coverage = model_df.set_index("Analysis")["Decision Coverage"]

ax = coverage.plot(
    kind="bar",
    figsize=(8, 5)
)

ax.set_title("Model Decision Coverage")
ax.set_ylabel("Decision Coverage")
ax.set_xlabel("Model")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_FOLDER, "decision_coverage.png"),
    dpi=300
)
plt.show()
plt.close()

comparison = df[df["Analysis"] == "SBERT-Qwen Agreement"]

if not comparison.empty:
    agreement = float(comparison.iloc[0]["Agreement Rate"])
    disagreement = 1 - agreement

    plt.figure(figsize=(7, 6))
    plt.pie(
        [agreement, disagreement],
        labels=["Agreement", "Disagreement"],
        autopct="%1.2f%%",
        startangle=90
    )
    plt.title("SBERT vs Qwen Prediction Agreement")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_FOLDER, "model_agreement.png"),
        dpi=300
    )
    plt.show()
    plt.close()

print("Visualization completed!")
print("Charts saved in:", OUTPUT_FOLDER)
