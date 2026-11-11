
import os
import pandas as pd
import matplotlib.pyplot as plt

METRICS_FILE = os.path.join(
    "results",
    "metrics_summary.csv"
)

OUTPUT_FOLDER = "results/graphs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(METRICS_FILE)

# 1. Performance Comparison
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

ax = df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title("Caption Error Detection - Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Model")
ax.legend(loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "performance_comparison.png"),
    dpi=300
)

plt.show()
plt.close()

# 2. Coverage Comparison
ax = df.set_index("Model")["Coverage"].plot(
    kind="bar",
    figsize=(8, 5)
)

ax.set_title("Model Coverage Comparison")
ax.set_ylabel("Coverage")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Model")
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "coverage_comparison.png"),
    dpi=300
)

plt.show()
plt.close()

print("Visualization completed!")
print("Charts saved in:", OUTPUT_FOLDER)
