
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CSV_FILE = "results/experiment_results.csv"
OUTPUT_FOLDER = "results/graphs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(CSV_FILE)

# Actual labels: ERROR = 1, CORRECT = 0
y_true = (
    df["Ground Truth"].str.upper() == "ERROR"
).astype(int)

models = {
    "Qwen VLM": lambda row: (
        1 if row["Qwen Verdict"] == "NOT_SUPPORTED"
        else 0 if row["Qwen Verdict"] == "SUPPORTED"
        else -1
    ),
    "Combined System": lambda row: (
        1 if row["Final Decision"] == "POSSIBLE CAPTION ERROR"
        else 0 if row["Final Decision"] == "CAPTION SUPPORTED"
        else -1
    )
}

for model_name, prediction_function in models.items():

    predictions = df.apply(prediction_function, axis=1)

    # Exclude NEEDS REVIEW
    valid = predictions != -1

    actual = y_true[valid]
    predicted = predictions[valid]

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["CORRECT", "ERROR"]
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"{filename}_confusion_matrix.png"
        ),
        dpi=300
    )

    plt.show()
    plt.close()

print("Confusion matrices generated!")
