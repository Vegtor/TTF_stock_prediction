import torch
import torch.nn as nn

class Gated_Residual_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        """
        Gated Residual Network (GRN) as used in Temporal Fusion Transformers (TFT).

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
            output_dim (int, optional): Desired output dimension.
                Defaults to None.
            dropout (float, optional): Dropout rate for regularization.
                Defaults to 0.1.
        """
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        self.gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        Forward pass through the Gated Residual Network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, window_size, output_dim)
        """
        residual = x if self.skip is None else self.skip(x)
        x = self.linear1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        gate = self.gate(x)
        x = gate * x + (1 - gate) * residual
        return self.norm(x)

class Temporal_Fusion_Transformer(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_heads=4, dropout=0.1):
        """
        A simplified version of the Temporal Fusion Transformer (TFT).
        This model combines gated residual networks, self-attention, and feedforward layers for time series modeling.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Size of the hidden representations.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.input_projection = nn.Linear(input_size, hidden_size)

        self.grn1 = Gated_Residual_Network(hidden_size, hidden_size)
        self.grn2 = Gated_Residual_Network(hidden_size, hidden_size)

        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.attn_layer_norm = nn.LayerNorm(hidden_size)

        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        self.ff_layer_norm = nn.LayerNorm(hidden_size)

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, *_):
        """
        Forward pass for the Temporal Fusion Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, 1)
        """
        x = self.input_projection(x)
        x = self.grn1(x)

        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.attn_layer_norm(x + attn_output)

        ff_output = self.positionwise_feedforward(x)
        x = self.ff_layer_norm(x + ff_output)

        x = self.grn2(x)

        return self.output_layer(x[:, -1:, :])
