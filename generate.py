"""Gera texto a partir de um checkpoint criado por train.py."""

import argparse
from pathlib import Path

import torch

from model import GPTLanguageModel

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Gera texto com um modelo treinado")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "checkpoints" / "model.pt"
    )
    parser.add_argument("--prompt", default="A biblioteca")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens não pode ser negativo")
    if args.temperature <= 0:
        parser.error("--temperature precisa ser maior que zero")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # O checkpoint é produzido por este projeto e contém pesos e opções simples.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    vocabulary = checkpoint["vocabulary"]
    stoi = {character: index for index, character in enumerate(vocabulary)}
    itos = {index: character for character, index in stoi.items()}

    unknown = sorted(set(args.prompt) - set(stoi))
    if unknown:
        parser.error(
            "o prompt contém caracteres ausentes no corpus de treino: "
            + repr("".join(unknown))
        )

    config = checkpoint["config"]
    model = GPTLanguageModel(
        vocab_size=len(vocabulary),
        block_size=config["block_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # Desliga dropout durante a geração.

    prompt_ids = [stoi[character] for character in args.prompt]
    if not prompt_ids:
        prompt_ids = [0]
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )[0].tolist()
    print("".join(itos[index] for index in generated))


if __name__ == "__main__":
    main()
