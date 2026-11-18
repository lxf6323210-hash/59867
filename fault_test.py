
from fault_injection import inject_fault

test_captions = [
    "A cat is sitting on a couch.",
    "A woman is standing near a building.",
    "A red car is parked beside a tree.",
    "A dog is playing in the garden.",
    "A computer is sitting on a table."
]

for caption in test_captions:
    result = inject_fault(caption)

    print("\nOriginal:", result["original_caption"])
    print("Modified:", result["modified_caption"])
    print("Changed:", result["changed_word"])
    print("Replacement:", result["replacement_word"])
    print("Success:", result["fault_injected"])
    print("Reason:", result["reason"])
