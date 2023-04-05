import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

random_seed = 42

class AutoencoderTransformer(nn.Module):
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

    def forward(self, x):
        # x has shape (batch_size, seq_len, input_dim)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, encoded)
        output = self.output_layer(torch.mean(encoded, dim=1))
        # output has shape (batch_size, output_dim)
        return output, decoded

model = AutoencoderTransformer(input_dim=5, hidden_dim=32, num_layers=2, num_heads=5, output_dim=32)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

fingerprints_dir = '../output/fingerprints'
models_dir = '../output/models'

fingerprints = []
for file in os.listdir(fingerprints_dir):
    fingerprints.append(np.loadtxt(os.path.join(fingerprints_dir, file)))

train_loader = [torch.from_numpy(x) for x in fingerprints]
train_loader = [torch.unsqueeze(x, 0) for x in train_loader]
train_loader = [x.to(torch.float) for x in train_loader]

for epoch in range(5):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs = data
        optimizer.zero_grad()
        outputs, decoded = model(inputs)
        loss = criterion(inputs, decoded)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20== 19:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

    torch.save(model, os.path.join(models_dir,f"AutoencoderTransformer_{epoch}.pt"))
