import torch
import torch.nn as nn

class SimpleFeedForward(nn.Module):
    def __init__(self, input_dim=100, params=None):
        super().__init__()
        if params is None:
            params = {}

        hidden1 = params.get('hidden1', 64)
        hidden2 = params.get('hidden2', 32)
        hidden3 = params.get('hidden3', 16)
        dropout_rate = params.get('dropout', 0.2)
        activation_name = params.get('activation', 'relu')

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            self.get_activation(activation_name),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden1, hidden2),
            self.get_activation(activation_name),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden2, hidden3),
            self.get_activation(activation_name),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden3, 1)
        )

    def get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'swish': nn.SiLU()
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        return torch.sigmoid(self.layers(x)).squeeze(1)

