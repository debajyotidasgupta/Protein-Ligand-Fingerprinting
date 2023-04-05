import torch
import torch.nn as nn

class AutoencoderTransformer(nn.Module):
    """
    A sequence to sequence self prediction transformer
    A sequence is fed as input to the transformer encoder
    The transformer decoder tries to obtain the original input by decoding
    The average of all encoder states is fed to a linear layer followed by sigmoid
    to obtain a fixed length embedding for the input sequence
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim

        # Define the TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Define the TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Define the output layer
        self.output_layer = nn.Linear(input_dim, output_dim)

        # Final sigmoid layer to  bring values between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x has shape (batch_size, seq_len, input_dim)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, encoded)
        output = self.output_layer(torch.mean(encoded, dim=1))
        output = self.sigmoid(output)
        # output has shape (batch_size, output_dim)
        return output, decoded

