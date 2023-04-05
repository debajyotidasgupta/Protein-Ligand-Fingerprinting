import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from fingerprint.transformer import *

fingerprints_dir = 'output/fingerprints'
models_dir = 'output/models'

def train(input_dim=5, hidden_dim=32, num_layers=2, num_heads=5, output_dim=256, epochs=5):
    model = AutoencoderTransformer(input_dim=input_dim, hidden_dim=hidden_dim, 
                                   num_layers=num_layers, num_heads=num_heads, output_dim=output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    fingerprints = []
    for file in os.listdir(fingerprints_dir):
        fingerprints.append(np.loadtxt(os.path.join(fingerprints_dir, file)))

    train_loader = [torch.from_numpy(x) for x in fingerprints]
    train_loader = [torch.unsqueeze(x, 0) for x in train_loader]
    train_loader = [x.to(torch.float) for x in train_loader]

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs = data
            optimizer.zero_grad()
            _, decoded = model(inputs)
            loss = criterion(inputs, decoded)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20== 19:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

        torch.save(model, os.path.join(models_dir,f"AutoencoderTransformer_{epoch}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dim')
    parser.add_argument('-h', '--hidden_dim')
    parser.add_argument('-l', '--num_layers')
    parser.add_argument('-nh', '--num_heads')
    parser.add_argument('-o', '--output_dim')
    parser.add_argument('-e', '--epochs')
    args = parser.parse_args()

    
