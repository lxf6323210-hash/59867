
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DISTRIBUTION_FILE = "results/prediction_distribution_3models.csv"
AGREEMENT_FILE = "results/visual_model_agreement.csv"
CROSS_TAB_FILE = "results/qwen_llama_cross_tabulation.csv"

GRAPH_DIR = "results/graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

distribution = pd.read_csv(DISTRIBUTION_FILE)
agreement = pd.read_csv(AGREEMENT_FILE)
cross_tab = pd.read_csv(CROSS_TAB_FILE, index_col=0)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

sbert = distribution[distribution["Model"] == "SBERT"].copy()

sbert = sbert[
    sbert["Prediction"].isin(["HIGH_SIMILARITY", "LOW_SIMILARITY"])
]

plt.figure()

bars = plt.bar(
    sbert["Prediction"],
    sbert["Count"]
)

plt.title("SBERT Text Similarity Distribution")
plt.xlabel("Similarity Category")
plt.ylabel("Number of Records")

for bar, count, percentage in zip(
    bars,
    sbert["Count"],
    sbert["Percentage"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(count)} ({percentage:.2f}%)",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    os.path.join(GRAPH_DIR, "sbert_similarity_distribution.png"),
    dpi=300
)
plt.close()

visual = distribution[
    distribution["Model"].isin(["Qwen", "Llama"])
].copy()

categories = ["SUPPORTED", "UNSUPPORTED", "UNCERTAIN"]

models = ["Qwen", "Llama"]
x = np.arange(len(categories))
width = 0.35

plt.figure()

for i, model in enumerate(models):
    model_data = visual[visual["Model"] == model].set_index("Prediction")
    counts = [
        int(model_data.loc[c, "Count"])
        if c in model_data.index else 0
        for c in categories
    ]

    positions = x + (i - 0.5) * width

    bars = plt.bar(
        positions,
        counts,
        width,
        label=model
    )

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom"
        )

plt.title("Qwen vs. Llama Prediction Distribution")
plt.xlabel("Visual Verification Prediction")
plt.ylabel("Number of Records")
plt.xticks(x, categories)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "visual_model_prediction_distribution.png"),
    dpi=300
)
plt.close()

agreement_row = agreement.iloc[0]

agreement_values = [
    int(agreement_row["Agreement"]),
    int(agreement_row["Disagreement"])
]

agreement_labels = [
    "Agreement",
    "Disagreement"
]

agreement_percentages = [
    value / int(agreement_row["Comparable Records"]) * 100
    for value in agreement_values
]

plt.figure()

bars = plt.bar(
    agreement_labels,
    agreement_percentages
)

plt.title("Qwen-Llama Model Agreement")
plt.xlabel("Comparison Result")
plt.ylabel("Percentage (%)")
plt.ylim(0, 100)

for bar, value, percentage in zip(
    bars,
    agreement_values,
    agreement_percentages
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value} ({percentage:.2f}%)",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "qwen_llama_agreement.png"),
    dpi=300
)
plt.close()

cross_tab = cross_tab.reindex(
    index=categories,
    columns=categories,
    fill_value=0
)

matrix = cross_tab.to_numpy()

plt.figure(figsize=(9, 7))

plt.imshow(matrix, interpolation="nearest", aspect="auto")

plt.title("Qwen vs. Llama Prediction Cross-Tabulation")
plt.xlabel("Llama Prediction")
plt.ylabel("Qwen Prediction")

plt.xticks(
    np.arange(len(categories)),
    categories,
    rotation=25,
    ha="right"
)

plt.yticks(
    np.arange(len(categories)),
    categories
)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        plt.text(
            j,
            i,
            str(matrix[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar(label="Number of Records")
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "qwen_llama_cross_tabulation.png"),
    dpi=300
)
plt.close()

print("\nVISUALIZATION COMPLETED")

print("\nGenerated graphs:")

for filename in [
    "sbert_similarity_distribution.png",
    "visual_model_prediction_distribution.png",
    "qwen_llama_agreement.png",
    "qwen_llama_cross_tabulation.png"
]:
    print(os.path.join(GRAPH_DIR, filename))

print("\nAll four graphs generated successfully.")
