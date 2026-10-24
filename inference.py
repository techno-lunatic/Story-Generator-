from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import LanguageModel


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "artifacts" / "final_model"
CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

# Model dimensions saved in the notebook's training configuration
EMBED_DIM = 256
N_HEADS = 4
N_LAYERS = 4
BLOCK_SIZE = 128
DROPOUT = 0.1


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}. "
            "Place best_model.pth in artifacts/final_model/."
        )
    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}. "
            "Place tokenizer.json in artifacts/final_model/."
        )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    head_size = EMBED_DIM // N_HEADS

    model = LanguageModel(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        head_size=head_size,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt, top_k, temperature, max_new_tokens):
    encoded_prompt = tokenizer.encode(prompt)
    context = torch.tensor(
        [encoded_prompt.ids],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        generated_ids = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    return tokenizer.decode(generated_ids[0].tolist())
