
import torch

from semantic_similarity import (
    load_sbert,
    compute_similarity
)

sbert = load_sbert()

caption1 = "A dog is playing in the park."

caption2 = "A puppy is running outside."

caption3 = "A red car is parked beside a building."

score1 = compute_similarity(
    caption1,
    caption2,
    sbert
)

score2 = compute_similarity(
    caption1,
    caption3,
    sbert
)

print("\n========== SBERT TEST ==========")

print("Caption 1:", caption1)
print("Caption 2:", caption2)
print("Similarity 1:", round(score1, 4))

print()

print("Caption 1:", caption1)
print("Caption 3:", caption3)
print("Similarity 2:", round(score2, 4))

print("================================")
