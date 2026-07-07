"""Cache MiniLM input embedding table to disk.

The encoder and decoder both load this frozen table to share a multilingual
vocabulary. Cached as a single torch tensor so we don't depend on
sentence-transformers at training time.
"""

import argparse
import os

import torch
from transformers import AutoModel


def cache_embedding_table(
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    output: str = "data/embedding_table.pt",
) -> None:
    print(f"Loading {model_name} to extract embedding table...")
    model = AutoModel.from_pretrained(model_name)
    emb = model.embeddings.word_embeddings.weight.detach().clone()
    print(f"  shape: {tuple(emb.shape)}, dtype: {emb.dtype}")
    print(f"  L2 norm range: {emb.norm(dim=1).min():.3f} - {emb.norm(dim=1).max():.3f}")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(emb, output)
    sz = os.path.getsize(output) / 1e6
    print(f"  saved to {output} ({sz:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/embedding_table.pt")
    args = parser.parse_args()
    cache_embedding_table(output=args.output)
