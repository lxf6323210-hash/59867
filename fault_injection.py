
import random
import re


# Words and their incorrect replacements.
FAULT_WORDS = {

    # Animals
    "cat": ["dog", "rabbit", "bird"],
    "dog": ["cat", "horse", "rabbit"],
    "bird": ["dog", "cat", "airplane"],
    "horse": ["cow", "dog", "zebra"],
    "cow": ["horse", "dog", "sheep"],
    "rabbit": ["cat", "dog", "bird"],
    "sheep": ["cow", "horse", "dog"],
    "elephant": ["horse", "cow", "giraffe"],
    "bear": ["dog", "horse", "lion"],
    "lion": ["dog", "cat", "horse"],
    "fish": ["bird", "cat", "dog"],

    # Vehicles
    "car": ["bus", "bicycle", "airplane"],
    "bus": ["car", "truck", "train"],
    "truck": ["car", "bus", "motorcycle"],
    "bicycle": ["motorcycle", "car", "bus"],
    "motorcycle": ["bicycle", "car", "truck"],
    "airplane": ["bird", "car", "helicopter"],
    "train": ["bus", "car", "airplane"],
    "boat": ["car", "bus", "airplane"],

    # Furniture and objects
    "couch": ["table", "bed", "chair"],
    "sofa": ["table", "bed", "chair"],
    "table": ["couch", "bed", "chair"],
    "chair": ["sofa", "table", "bed"],
    "bed": ["table", "sofa", "chair"],
    "pillow": ["tree", "window", "book"],
    "window": ["door", "mirror", "painting"],
    "book": ["phone", "bottle", "shoe"],
    "phone": ["book", "camera", "remote"],
    "bottle": ["cup", "book", "shoe"],
    "cup": ["bottle", "plate", "book"],
    "lamp": ["fan", "clock", "television"],
    "clock": ["lamp", "camera", "phone"],

    # People and sports
    "person": ["dog", "robot", "statue"],
    "people": ["dogs", "robots", "statues"],
    "man": ["woman", "dog", "robot"],
    "woman": ["man", "dog", "robot"],
    "player": ["dog", "robot", "car"],
    "athlete": ["robot", "dog", "horse"],
    "ball": ["book", "bottle", "shoe"],
    "basketball": ["soccer ball", "football", "book"],
    "football": ["basketball", "book", "bottle"],
    "soccer": ["basketball", "tennis", "swimming"],
    "tennis": ["soccer", "basketball", "baseball"],
    "baseball": ["tennis", "soccer", "basketball"],
    "skateboard": ["bicycle", "car", "table"],

    # Buildings and landscapes
    "tree": ["building", "car", "chair"],
    "building": ["tree", "car", "mountain"],
    "house": ["car", "tree", "boat"],
    "mountain": ["building", "tree", "ocean"],
    "river": ["road", "building", "forest"],
    "ocean": ["desert", "road", "forest"],
    "beach": ["mountain", "street", "forest"],
    "forest": ["city", "desert", "ocean"],
    "street": ["beach", "river", "forest"],
    "road": ["river", "beach", "forest"],
    "sky": ["ocean", "grass", "road"],
    "grass": ["water", "sand", "snow"],
    "landscape": ["city", "ocean", "desert"],
    "field": ["ocean", "street", "building"],

    # Food
    "pizza": ["hamburger", "cake", "sandwich"],
    "hamburger": ["pizza", "salad", "cake"],
    "sandwich": ["pizza", "hamburger", "apple"],
    "cake": ["pizza", "bread", "salad"],
    "apple": ["orange", "banana", "carrot"],
    "banana": ["apple", "orange", "tomato"],
    "bread": ["pizza", "cake", "sandwich"],
    "food": ["book", "phone", "shoe"],
}


def inject_fault(caption, seed=None):

    """
    Replace one recognizable word in a caption
    to create a synthetic caption error.
    """

    rng = random.Random(seed)

    candidates = []

    # Find all replaceable words.
    for word in FAULT_WORDS:

        pattern = rf"\b{re.escape(word)}\b"

        if re.search(pattern, caption, flags=re.IGNORECASE):
            candidates.append(word)

    if not candidates:

        return {
            "original_caption": caption,
            "modified_caption": caption,
            "changed_word": None,
            "replacement_word": None,
            "fault_injected": False,
            "reason": "No replaceable word found in FAULT_WORDS."
        }

    # Randomly select one word.
    selected_word = rng.choice(candidates)

    replacement_word = rng.choice(
        FAULT_WORDS[selected_word]
    )

    pattern = rf"\b{re.escape(selected_word)}\b"

    # Preserve capitalization.
    def replace_match(match):

        original = match.group(0)

        if original.isupper():
            return replacement_word.upper()

        if original.istitle():
            return replacement_word.title()

        return replacement_word

    modified_caption = re.sub(
        pattern,
        replace_match,
        caption,
        count=1,
        flags=re.IGNORECASE
    )

    return {
        "original_caption": caption,
        "modified_caption": modified_caption,
        "changed_word": selected_word,
        "replacement_word": replacement_word,
        "fault_injected": True,
        "reason": None
    }
