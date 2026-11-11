
import torch
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"


def load_sbert(device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Sentence-BERT...")
    print("Device:", device)

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    print("Sentence-BERT loaded successfully!")

    return model


def compute_similarity(c1, c2, model):

    emb1 = model.encode(
        c1,
        convert_to_tensor=True
    )

    emb2 = model.encode(
        c2,
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        emb1,
        emb2
    ).item()

    return similarity
