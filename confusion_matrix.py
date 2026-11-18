
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = Path("results/experiment_results_200_with_llama.csv")
RESULTS_DIR = Path("results")
GRAPH_DIR = RESULTS_DIR / "graphs"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["SUPPORTED", "UNSUPPORTED", "UNCERTAIN"]


def normalize_verdict(value):
    value = str(value).strip().upper()

    if value in ["SUPPORTED", "CORRECT", "TRUE", "YES"]:
        return "SUPPORTED"

    if value in [
        "NOT_SUPPORTED",
        "UNSUPPORTED",
        "ERROR",
        "INCORRECT",
        "FALSE",
        "NO"
    ]:
        return "UNSUPPORTED"

    if value in ["UNCERTAIN", "UNKNOWN", "REVIEW"]:
        return "UNCERTAIN"

    return "UNCERTAIN"


def create_cross_tabulation(
    df,
    model_a,
    model_b,
    column_a,
    column_b
):
    matrix = pd.crosstab(
        df[column_a],
        df[column_b]
    )

    matrix = matrix.reindex(
        index=LABELS,
        columns=LABELS,
        fill_value=0
    )

    csv_path = RESULTS_DIR / (
        f"model_cross_tabulation_{model_a.lower()}_{model_b.lower()}.csv"
    )

    matrix.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(matrix.values, aspect="auto")

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))

    ax.set_xticklabels(LABELS, rotation=20)
    ax.set_yticklabels(LABELS)

    ax.set_xlabel(f"{model_b} Prediction")
    ax.set_ylabel(f"{model_a} Prediction")

    ax.set_title(
        f"{model_a} vs {model_b} Prediction Cross-Tabulation"
    )

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(
                j,
                i,
                str(matrix.iloc[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax, label="Number of Records")
    fig.tight_layout()

    image_path = GRAPH_DIR / (
        f"model_cross_tabulation_{model_a.lower()}_{model_b.lower()}.png"
    )

    fig.savefig(image_path, dpi=300)
    plt.close(fig)

    print("\n", model_a, "vs", model_b)
    print(matrix)
    print("Saved:", csv_path)
    print("Saved:", image_path)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "Similarity",
        "Qwen Verdict",
        "Llama Verdict"
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Similarity"] = pd.to_numeric(
        df["Similarity"],
        errors="coerce"
    )

    if df["Similarity"].isna().any():
        raise ValueError("Invalid Similarity values found.")

    df["SBERT Prediction"] = df["Similarity"].apply(
        lambda value: "SUPPORTED"
        if value >= 0.50
        else "UNSUPPORTED"
    )

    df["Qwen Prediction"] = df["Qwen Verdict"].apply(
        normalize_verdict
    )

    df["Llama Prediction"] = df["Llama Verdict"].apply(
        normalize_verdict
    )

    create_cross_tabulation(
        df,
        "SBERT",
        "Qwen",
        "SBERT Prediction",
        "Qwen Prediction"
    )

    create_cross_tabulation(
        df,
        "SBERT",
        "Llama",
        "SBERT Prediction",
        "Llama Prediction"
    )

    create_cross_tabulation(
        df,
        "Qwen",
        "Llama",
        "Qwen Prediction",
        "Llama Prediction"
    )

    print("\nAll prediction cross-tabulations completed.")


if __name__ == "__main__":
    main()
