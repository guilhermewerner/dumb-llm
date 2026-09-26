"""Um GPT minúsculo, montado com módulos básicos do PyTorch."""

import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    """Uma cabeça de self-attention causal (scaled dot-product attention)."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # A máscara impede que uma posição veja caracteres futuros.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        key = self.key(x)
        query = self.query(x)

        # Cada posição mede quanto deve prestar atenção às posições anteriores.
        weights = query @ key.transpose(-2, -1) * (key.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        value = self.value(x)
        return weights @ value


class MultiHeadAttention(nn.Module):
    """Várias cabeças de atenção aprendem relações em paralelo."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd precisa ser divisível por n_head")
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(joined))


class FeedForward(nn.Module):
    """Pequena rede que processa cada posição depois da atenção."""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Block(nn.Module):
    """Um bloco Transformer: atenção e rede feed-forward com atalhos."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conexões residuais ajudam o gradiente a atravessar blocos sucessivos.
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Modelo de linguagem que prevê o próximo caractere em cada posição."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 64,
        n_embd: int = 64,
        n_head: int = 4,
        n_layer: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.final_norm = nn.LayerNorm(n_embd)
        self.language_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, time = idx.shape
        if time > self.block_size:
            raise ValueError("A sequência é maior que block_size")

        positions = torch.arange(time, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.language_head(self.final_norm(x))

        loss = None
        if targets is not None:
            # Cross entropy compara a distribuição prevista com o próximo token real.
            loss = F.cross_entropy(
                logits.view(batch * time, -1), targets.view(batch * time)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ):
        """Gera tokens por amostragem; temperatura regula a aleatoriedade."""
        if temperature <= 0:
            raise ValueError("temperature precisa ser maior que zero")
        for _ in range(max_new_tokens):
            # O Transformer só aceita block_size caracteres de contexto.
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
