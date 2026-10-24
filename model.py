import torch
from torch.nn import functional as F
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, embed_dim, head_size, block_size, dropout):
        super().__init__()

        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        # mask prevents attention to future tokens
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Generate query, key, and value vectors
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Calculate scaled attention scores
        scores = q @ k.transpose(-2, -1)
        scores = scores * (k.shape[-1] ** -0.5)

        # Prevent each token from looking at future tokens
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        # Convert scores into attention probabilities
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Weighted combination of value vectors
        out = weights @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, head_size, block_size, dropout):
        super().__init__()

        # Raise an error if the dimensions are incompatible
        if embed_dim != n_heads * head_size:
            raise ValueError(
                f"embed_dim ({embed_dim}) must equal "
                f"n_heads ({n_heads}) * head_size ({head_size})"
            )

        # Create multiple independent attention heads
        self.heads = nn.ModuleList([
            Head(embed_dim, head_size, block_size, dropout)
            for _ in range(n_heads)
        ])

        # Concatenate the heads' outputs and mix their information
        self.output_projection = nn.Linear(
            n_heads * head_size,
            embed_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run each attention head
        head_outputs = [head(x) for head in self.heads]

        # Combine the outputs along the feature dimension
        out = torch.cat(head_outputs, dim=-1)

        # Mix information from all heads
        out = self.output_projection(out)

        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()

        self.network = nn.Sequential(
            # Expand the representation
            nn.Linear(embed_dim, 4 * embed_dim),

            # Introduce non-linearity
            nn.GELU(),

            # Return to the original embedding dimension
            nn.Linear(4 * embed_dim, embed_dim),

            # Regularization during training
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, dropout):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T = x.shape

        token_emb = self.token_embedding(x)

        positions = torch.arange(T, device=x.device)
        position_emb = self.position_embedding(positions)

        out = token_emb + position_emb
        out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_dim, n_heads, head_size,
        block_size, dropout
    ):
        super().__init__()

        # Multi-head self-attention
        self.attention = MultiHeadAttention(
            embed_dim, n_heads, head_size, block_size, dropout
        )

        # Feed-forward network
        self.feed_forward = FeedForward(embed_dim, dropout)

        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Attention sublayer with residual connection
        x = x + self.attention(self.ln1(x))

        # Feed-forward sublayer with residual connection
        x = x + self.feed_forward(self.ln2(x))

        return x


class LanguageModel(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, n_heads,
        n_layers, head_size, block_size, dropout
    ):
        super().__init__()

        self.embedding = TokenAndPositionEmbedding(
            vocab_size, embed_dim, block_size, dropout
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, n_heads, head_size,
                block_size, dropout
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size

    def forward(self, x, targets=None):
        B, T = x.shape

        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                targets.reshape(B * T)
            )

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            context_window = idx[:, -self.block_size:]

            logits, loss = self(context_window)
            logits = logits[:, -1, :]

            # Applying Temperature
            logits = logits / temperature

            # Implementing top_k
            if top_k is not None:
                k = min(top_k, logits.size(-1))

                top_k_values, _ = torch.topk(logits, k, dim=-1)
                cutoff = top_k_values[:, -1].unsqueeze(-1)

                logits[logits < cutoff] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
