import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomizableNicon(nn.Module):
    def __init__(self, input_channels=1, params=None):
        super().__init__()
        if params is None:
            params = {}

        self.params = params  # store for later use
        self.spatial_dropout = nn.Dropout2d(params.get('spatial_dropout', 0.08))

        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=params.get('filters1', 8),
            kernel_size=params.get('kernel_size1', 15),
            stride=params.get('strides1', 5)
        )
        self.activation1 = self.get_activation("selu")
        self.dropout = nn.Dropout(params.get('dropout_rate', 0.2))

        self.conv2 = nn.Conv1d(
            in_channels=params.get('filters1', 8),
            out_channels=params.get('filters2', 64),
            kernel_size=params.get('kernel_size2', 21),
            stride=params.get('strides2', 3)
        )
        self.norm1 = nn.BatchNorm1d(params.get('filters2', 64))
        self.activation2 = self.get_activation("relu")

        self.conv3 = nn.Conv1d(
            in_channels=params.get('filters2', 64),
            out_channels=params.get('filters3', 32),
            kernel_size=params.get('kernel_size3', 5),
            stride=params.get('strides3', 3)
        )
        self.norm2 = nn.BatchNorm1d(params.get('filters3', 32))
        self.activation3 = self.get_activation("elu")

        self.flatten = nn.Flatten()
        self.dense_units = params.get('dense_units', 16)
        self.dense_activation = self.get_activation("sigmoid")

        self.dense = nn.LazyLinear(self.dense_units)
        self.out = nn.LazyLinear(params.get('output_dim', 1))

    def get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'swish': nn.SiLU()
        }
        return activations.get(name.lower(), nn.ReLU())
        
    def get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'swish': nn.SiLU()
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x):
        if x.device != next(self.parameters()).device:
            raise RuntimeError(
                f"[ERROR] Input device mismatch: x is on {x.device} but model is on {next(self.parameters()).device}"
            )
        try:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            x = self.spatial_dropout(x)
            if x.shape[-1] < self.conv1.kernel_size[0]:
                pad = self.conv1.kernel_size[0] - x.shape[-1]
                x = F.pad(x, (pad, 0))
            x = self.activation1(self.conv1(x))
            x = self.dropout(x)
            if x.shape[-1] < self.conv2.kernel_size[0]:
                pad = self.conv2.kernel_size[0] - x.shape[-1]
                x = F.pad(x, (pad, 0))
            x = self.conv2(x)

            if isinstance(self.norm1, nn.LayerNorm):
                x = x.permute(0, 2, 1)
                x = self.norm1(x)
                x = x.permute(0, 2, 1)
            else:
                x = self.norm1(x)
            
            x = self.activation2(x)
            if x.shape[-1] < self.conv3.kernel_size[0]:
                pad = self.conv3.kernel_size[0] - x.shape[-1]
                x = F.pad(x, (pad, 0))
            x = self.conv3(x)
            
            if isinstance(self.norm2, nn.LayerNorm):
                x = x.permute(0, 2, 1)
                x = self.norm2(x)
                x = x.permute(0, 2, 1)
            else:
                x = self.norm2(x)
            
            x = self.activation3(x)
            x = self.flatten(x)
            x = self.dense_activation(self.dense(x))
            
            x = torch.sigmoid(self.out(x))
            return x

        except RuntimeError as e:
            print("\n[RuntimeError in CustomizableNicon.forward()]")
            print(f"Exception: {e}")
            print(f"x device: {x.device}, x dtype: {x.dtype}, x shape: {x.shape}")
            for name, param in self.named_parameters():
                print(f"{name}: {param.device}, {param.dtype}")
            raise e  # Re-raise for visibility in training loop