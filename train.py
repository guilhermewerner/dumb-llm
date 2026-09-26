"""Treina um pequeno modelo GPT sobre um arquivo de texto."""

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from model import GPTLanguageModel

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # Valores pequenos para o modelo caber em computadores comuns.
    batch_size: int = 32
    block_size: int = 64
    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20
    learning_rate: float = 3e-4
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    seed: int = 1337


def read_corpus(path: Path) -> tuple[str, list[str]]:
    """Lê texto UTF-8 e cria um vocabulário ordenado de caracteres."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"O arquivo está vazio: {path}")
    vocabulary = sorted(set(text))
    return text, vocabulary


def get_batch(data: torch.Tensor, batch_size: int, block_size: int):
    """Sorteia vários trechos e seus alvos (o mesmo trecho adiantado em 1)."""
    starts = torch.randint(
        len(data) - block_size - 1, (batch_size,), device=data.device
    )
    offsets = torch.arange(block_size, device=data.device)
    x = data[starts[:, None] + offsets]
    y = data[starts[:, None] + offsets + 1]
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device):
    """Estima a perda em alguns lotes sem calcular gradientes."""
    was_training = model.training
    model.eval()  # Desliga dropout para medir o modelo de forma estável.
    result = {}
    for name, data in (("treino", train_data), ("validacao", val_data)):
        losses = torch.zeros(config.eval_iters)
        for i in range(config.eval_iters):
            x, y = get_batch(data, config.batch_size, config.block_size)
            _, loss = model(x, y)
            losses[i] = loss.item()
        result[name] = losses.mean().item()
    model.train(was_training)
    return result


def main():
    parser = argparse.ArgumentParser(description="Treina um GPT de caracteres didático")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sample.txt")
    parser.add_argument("--max-iters", type=int, help="substitui Config.max_iters")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "checkpoints" / "model.pt"
    )
    args = parser.parse_args()

    config = Config()
    if args.max_iters is not None:
        if args.max_iters < 1:
            parser.error("--max-iters precisa ser pelo menos 1")
        config.max_iters = args.max_iters

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    text, vocabulary = read_corpus(args.data)
    stoi = {character: index for index, character in enumerate(vocabulary)}
    encoded = torch.tensor([stoi[character] for character in text], dtype=torch.long)
    split_at = int(0.9 * len(encoded))
    train_data = encoded[:split_at].to(device)
    val_data = encoded[split_at:].to(device)

    minimum = config.block_size + 2
    if len(train_data) < minimum or len(val_data) < minimum:
        raise ValueError(
            f"Use um corpus maior: treino e validação precisam ter ao menos {minimum} caracteres."
        )
    print(
        f"Corpus: {len(text):,} caracteres | vocabulário: {len(vocabulary)} caracteres"
    )

    model = GPTLanguageModel(
        vocab_size=len(vocabulary),
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parâmetros treináveis: {parameter_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    for step in range(config.max_iters):
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config, device)
            print(
                f"passo {step:4d} | perda treino {losses['treino']:.4f} "
                f"| validação {losses['validacao']:.4f}"
            )

        x, y = get_batch(train_data, config.batch_size, config.block_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # Calcula como cada peso contribuiu para o erro.
        optimizer.step()  # AdamW aplica uma atualização nos pesos.

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "vocabulary": vocabulary,
        },
        args.output,
    )
    print(f"Checkpoint salvo em: {args.output}")


if __name__ == "__main__":
    main()
