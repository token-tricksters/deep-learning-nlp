import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(input_size, input_size)
        self.v = nn.Linear(input_size, 1, bias=False)

    def forward(self, embeddings):
        # Apply linear transformation to the embeddings
        transformed = torch.tanh(self.W(embeddings))

        # Calculate attention weights
        attention_weights = torch.softmax(self.v(transformed), dim=1)

        # Apply attention weights to the embeddings
        attended_embeddings = torch.sum(attention_weights * embeddings, dim=1)

        return attended_embeddings
