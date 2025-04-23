import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by number of heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        # Get the length of the input sequences
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(queries)  # (N, query_len, heads, head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how pytorch does it
        # (N, query_len, heads, head_dim) matmul (N, key_len, heads, head_dim)
        # --> (N, heads, query_len, key_len) take transpose last two dims
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # queries(Q), keys(K)

        # Mask padded values so they don't get attention
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy(attention scores) across the last dimension
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)

        # (N, heads, query_len, key_len) matmul (N, value_len, heads, head_dim)
        # --> (N, query_len, heads, head_dim) take transpose last two dims
        out = torch.einsum("nhqv,nvhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # After concatting the heads in the last dimension we send this through a linear layer
        # to get the fully connected output
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add and norm
        x = self.dropout(self.norm1(attention + query))

        # Feed forward then add and norm
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class RadiationShieldingModel(nn.Module):
    def __init__(self, input_size, embed_size, heads, dropout, forward_expansion, num_layers, output_size):
        super(RadiationShieldingModel, self).__init__()
        self.embedding = nn.Linear(input_size, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout=dropout, forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        embedding = self.dropout(self.embedding(x))
        # Assuming input x is the same for value, key, and query for simplicity
        for layer in self.layers:
            embedding = layer(embedding, embedding, embedding, mask)

        out = self.fc_out(embedding)
        return out

if __name__ == "__main__":
    # Example usage:
    input_size = 3  # Example: Shielding weight, material density, distance
    output_size = 1 # Example: Radiation dose level
    embed_size = 64
    heads = 8
    dropout = 0.1
    forward_expansion = 4
    num_layers = 2
    seq_length = 1 # Assuming each input is a single data point

    model = RadiationShieldingModel(
        input_size, embed_size, heads, dropout, forward_expansion, num_layers, output_size
    )

    # Example input data (batch_size, seq_length, input_size)
    input_data = torch.randn((32, seq_length, input_size))
    mask = None # No padding in this simple example

    output = model(input_data, mask)
    print("Output shape:", output.shape) # Expected: (32, seq_length, output_size)
