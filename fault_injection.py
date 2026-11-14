
import random
import re

FAULT_WORDS = {
    "cat": ["dog", "rabbit", "fox"],
    "cats": ["dogs", "rabbits", "foxes"],
    "dog": ["cat", "horse", "rabbit"],
    "dogs": ["cats", "horses", "rabbits"],
    "bird": ["dog", "cat", "airplane"],
    "birds": ["dogs", "cats", "airplanes"],
    "horse": ["cow", "dog", "zebra"],
    "horses": ["cows", "dogs", "zebras"],
    "cow": ["horse", "dog", "sheep"],
    "cows": ["horses", "dogs", "sheep"],
    "rabbit": ["cat", "dog", "bird"],
    "rabbits": ["cats", "dogs", "birds"],
    "sheep": ["cow", "horse", "dog"],
    "elephant": ["horse", "cow", "giraffe"],
    "elephants": ["horses", "cows", "giraffes"],
    "bear": ["dog", "horse", "lion"],
    "bears": ["dogs", "horses", "lions"],
    "lion": ["dog", "cat", "horse"],
    "lions": ["dogs", "cats", "horses"],
    "fish": ["bird", "cat", "dog"],
    "dolphin": ["shark", "whale", "fish"],
    "dolphins": ["sharks", "whales", "fish"],
    "whale": ["dolphin", "shark", "fish"],
    "shark": ["dolphin", "whale", "fish"],
    "butterfly": ["bird", "bee", "dragonfly"],
    "butterflies": ["birds", "bees", "dragonflies"],
    "bee": ["butterfly", "bird", "fly"],
    "dragonfly": ["butterfly", "bee", "bird"],
    "flamingo": ["stork", "crane", "heron"],
    "flamingos": ["storks", "cranes", "herons"],
    "flaming": ["stork", "crane", "heron"],
    "flamings": ["storks", "cranes", "herons"],
    "giraffe": ["horse", "elephant", "camel"],
    "zebra": ["horse", "cow", "giraffe"],
    "car": ["bus", "bicycle", "airplane"],
    "cars": ["buses", "bicycles", "airplanes"],
    "bus": ["car", "truck", "train"],
    "buses": ["cars", "trucks", "trains"],
    "truck": ["car", "bus", "motorcycle"],
    "trucks": ["cars", "buses", "motorcycles"],
    "bicycle": ["motorcycle", "car", "bus"],
    "bicycles": ["motorcycles", "cars", "buses"],
    "motorcycle": ["bicycle", "car", "truck"],
    "motorcycles": ["bicycles", "cars", "trucks"],
    "motorbike": ["bicycle", "car", "truck"],
    "motorbikes": ["bicycles", "cars", "trucks"],
    "airplane": ["bird", "car", "helicopter"],
    "airplanes": ["birds", "cars", "helicopters"],
    "plane": ["bird", "car", "helicopter"],
    "planes": ["birds", "cars", "helicopters"],
    "train": ["bus", "car", "airplane"],
    "trains": ["buses", "cars", "airplanes"],
    "boat": ["car", "bus", "airplane"],
    "boats": ["cars", "buses", "airplanes"],
    "helicopter": ["airplane", "bird", "car"],
    "couch": ["table", "bed", "chair"],
    "sofa": ["table", "bed", "chair"],
    "table": ["couch", "bed", "chair"],
    "tables": ["couches", "beds", "chairs"],
    "chair": ["sofa", "table", "bed"],
    "chairs": ["sofas", "tables", "beds"],
    "bed": ["table", "sofa", "chair"],
    "beds": ["tables", "sofas", "chairs"],
    "pillow": ["tree", "window", "book"],
    "pillows": ["trees", "windows", "books"],
    "window": ["door", "mirror", "painting"],
    "windows": ["doors", "mirrors", "paintings"],
    "book": ["phone", "bottle", "shoe"],
    "books": ["phones", "bottles", "shoes"],
    "phone": ["book", "camera", "remote"],
    "phones": ["books", "cameras", "remotes"],
    "bottle": ["cup", "book", "shoe"],
    "bottles": ["cups", "books", "shoes"],
    "cup": ["bottle", "plate", "book"],
    "cups": ["bottles", "plates", "books"],
    "camera": ["phone", "clock", "television"],
    "cameras": ["phones", "clocks", "televisions"],
    "photograph": ["painting", "book", "map"],
    "lamp": ["fan", "clock", "television"],
    "clock": ["lamp", "camera", "phone"],
    "object": ["animal", "vehicle", "building"],
    "person": ["dog", "robot", "statue"],
    "people": ["dogs", "robots", "statues"],
    "man": ["woman", "dog", "robot"],
    "men": ["women", "dogs", "robots"],
    "woman": ["man", "dog", "robot"],
    "women": ["men", "dogs", "robots"],
    "player": ["dog", "robot", "car"],
    "athlete": ["robot", "dog", "horse"],
    "ball": ["book", "bottle", "shoe"],
    "basketball": ["soccer", "football", "book"],
    "football": ["basketball", "book", "bottle"],
    "soccer": ["basketball", "tennis", "swimming"],
    "tennis": ["soccer", "basketball", "baseball"],
    "baseball": ["tennis", "soccer", "basketball"],
    "skateboard": ["bicycle", "car", "table"],
    "tree": ["building", "car", "chair"],
    "trees": ["buildings", "cars", "chairs"],
    "building": ["tree", "car", "mountain"],
    "buildings": ["trees", "cars", "mountains"],
    "house": ["car", "tree", "boat"],
    "houses": ["cars", "trees", "boats"],
    "mountain": ["building", "tree", "ocean"],
    "mountains": ["buildings", "trees", "oceans"],
    "river": ["road", "building", "forest"],
    "rivers": ["roads", "buildings", "forests"],
    "ocean": ["desert", "road", "forest"],
    "oceans": ["deserts", "roads", "forests"],
    "beach": ["mountain", "street", "forest"],
    "beaches": ["mountains", "streets", "forests"],
    "forest": ["city", "desert", "ocean"],
    "forests": ["cities", "deserts", "oceans"],
    "street": ["beach", "river", "forest"],
    "streets": ["beaches", "rivers", "forests"],
    "road": ["river", "beach", "forest"],
    "roads": ["rivers", "beaches", "forests"],
    "sky": ["ocean", "grass", "road"],
    "grass": ["water", "sand", "snow"],
    "water": ["grass", "sand", "snow"],
    "landscape": ["city", "ocean", "desert"],
    "field": ["ocean", "street", "building"],
    "fields": ["oceans", "streets", "buildings"],
    "rock": ["tree", "car", "building"],
    "rocks": ["trees", "cars", "buildings"],
    "ice": ["sand", "grass", "water"],
    "pizza": ["hamburger", "cake", "sandwich"],
    "hamburger": ["pizza", "salad", "cake"],
    "sandwich": ["pizza", "hamburger", "apple"],
    "cake": ["pizza", "bread", "salad"],
    "apple": ["orange", "banana", "carrot"],
    "banana": ["apple", "orange", "tomato"],
    "bread": ["pizza", "cake", "sandwich"],
    "food": ["book", "phone", "shoe"],
}


def preserve_case(original, replacement):
    if original.isupper():
        return replacement.upper()

    if original.istitle():
        return replacement.title()

    return replacement


def fix_article(caption):
    pattern = r"\b(a|an)\s+([a-zA-Z]+)"

    def replace_article(match):
        article = match.group(1)
        word = match.group(2)

        if word[0].lower() in "aeiou":
            correct_article = "an"
        else:
            correct_article = "a"

        if article[0].isupper():
            correct_article = correct_article.capitalize()

        return f"{correct_article} {word}"

    return re.sub(
        pattern,
        replace_article,
        caption,
        flags=re.IGNORECASE
    )


# Multi-word phrases are checked before single words.
# These are candidate counterfactual substitutions, not guaranteed errors.
FAULT_PHRASES = {
    "passenger jet": ["passenger train", "helicopter", "sailboat"],
    "airport runway": ["swimming pool", "forest road", "train station"],
    "parking lot": ["swimming pool", "forest", "beach"],
    "universal mount": ["kitchen table", "wooden chair", "desk lamp"],
}

EXTRA_FAULT_WORDS = {
    "jet": ["train", "helicopter", "boat"],
    "runway": ["railway", "river", "beach"],
    "mug": ["bottle", "plate", "vase"],
    "fan": ["lamp", "clock", "television"],
    "motor": ["camera", "clock", "lamp"],
    "gopro": ["microscope", "telescope", "radio"],
    "canon": ["nikon", "sony", "fuji"],
    "olympus": ["nikon", "sony", "canon"],
    "fuji": ["nikon", "canon", "olympus"],
    "eos": ["lens", "screen", "button"],
    "akron": ["boston", "chicago", "seattle"],
    "hello": ["blue", "red", "green"],
    "kitty": ["puppy", "rabbit", "teddy"],
    "spa": ["school", "hotel", "office"],
}

ALL_FAULT_WORDS = dict(FAULT_WORDS)
ALL_FAULT_WORDS.update(EXTRA_FAULT_WORDS)


def assess_caption_quality(caption):
    """Reject obviously degenerate/repetitive captions."""
    if not isinstance(caption, str) or not caption.strip():
        return False, "Caption is empty or invalid."

    tokens = re.findall(r"[A-Za-z]+", caption.lower())
    if len(tokens) < 2:
        return False, "Caption is too short."

    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    most_repeated = max(counts.values(), default=0)
    repetition_ratio = most_repeated / len(tokens)

    if len(tokens) >= 5 and most_repeated >= 5 and repetition_ratio >= 0.40:
        repeated_token = max(counts, key=counts.get)
        return False, (
            f"LOW_QUALITY: token '{repeated_token}' is repeated "
            f"{most_repeated}/{len(tokens)} times."
        )

    if len(tokens) >= 6 and len(set(tokens)) <= 3:
        return False, "LOW_QUALITY: caption contains too few unique words."

    return True, None


def _find_matches(caption):
    """Find phrase matches first; otherwise find single-word matches."""
    phrase_matches = []
    for phrase in FAULT_PHRASES:
        pattern = rf"(?<![A-Za-z]){re.escape(phrase)}(?![A-Za-z])"
        match = re.search(pattern, caption, flags=re.IGNORECASE)
        if match:
            phrase_matches.append(
                (match.start(), match.end(), phrase, "phrase")
            )

    if phrase_matches:
        phrase_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        return phrase_matches

    word_matches = []
    for word in ALL_FAULT_WORDS:
        pattern = rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])"
        match = re.search(pattern, caption, flags=re.IGNORECASE)
        if match:
            word_matches.append((match.start(), match.end(), word, "word"))

    word_matches.sort(key=lambda x: x[0])
    return word_matches


def inject_fault(caption, seed=None):
    """
    Inject one candidate word/phrase substitution.

    A successful injection only means the text changed. Manually inspect
    the image before assigning Ground Truth = ERROR.
    """
    rng = random.Random(seed)

    usable, quality_reason = assess_caption_quality(caption)
    if not usable:
        return {
            "original_caption": caption,
            "modified_caption": caption,
            "changed_word": None,
            "replacement_word": None,
            "fault_injected": False,
            "reason": quality_reason,
            "low_quality": True
        }

    matches = _find_matches(caption)
    if not matches:
        return {
            "original_caption": caption,
            "modified_caption": caption,
            "changed_word": None,
            "replacement_word": None,
            "fault_injected": False,
            "reason": "No matching word or phrase found in fault vocabulary.",
            "low_quality": False
        }

    selected = rng.choice(matches)
    start, end, selected_term, match_type = selected

    if match_type == "phrase":
        replacement_pool = FAULT_PHRASES[selected_term]
    else:
        replacement_pool = ALL_FAULT_WORDS[selected_term]

    replacement = rng.choice(replacement_pool)
    original_text = caption[start:end]
    replacement_text = preserve_case(original_text, replacement)

    modified_caption = caption[:start] + replacement_text + caption[end:]
    modified_caption = fix_article(modified_caption)

    if modified_caption.strip().lower() == caption.strip().lower():
        return {
            "original_caption": caption,
            "modified_caption": caption,
            "changed_word": selected_term,
            "replacement_word": replacement,
            "fault_injected": False,
            "reason": "Replacement did not change the caption.",
            "low_quality": False
        }

    return {
        "original_caption": caption,
        "modified_caption": modified_caption,
        "changed_word": selected_term,
        "replacement_word": replacement,
        "fault_injected": True,
        "reason": None,
        "low_quality": False
    }
