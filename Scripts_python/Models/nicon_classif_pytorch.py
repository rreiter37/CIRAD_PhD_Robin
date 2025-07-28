import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "swish":
        return nn.SiLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softmax":
        return nn.Softmax(dim=1)
    else:
        raise ValueError(f"Unknown activation function: {name}")

class SpatialDropout1D(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (B, C, 1, L)
        x = super().forward(x)
        return x.squeeze(2)  # (B, C, L)

class CustomNiconClassifier(nn.Module):
    def __init__(self, input_shape, num_classes=2, params={}):
        super().__init__()
        in_channels = input_shape[-1]
        self.spatial_dropout = SpatialDropout1D(params.get("spatial_dropout", 0.08))

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=params.get("filters1", 8),
                               kernel_size=params.get("kernel_size1", 15),
                               stride=params.get("strides1", 5))
        self.act1 = get_activation(params.get("activation1", "selu"))
        self.dropout1 = nn.Dropout(params.get("dropout_rate", 0.2))

        self.conv2 = nn.Conv1d(in_channels=params.get("filters1", 8),
                               out_channels=params.get("filters2", 64),
                               kernel_size=params.get("kernel_size2", 21),
                               stride=params.get("strides2", 3))
        self.act2 = get_activation(params.get("activation2", "relu"))
        self.norm1 = nn.BatchNorm1d(params.get("filters2", 64)) if params.get("normalization_method1", "BatchNormalization") == "BatchNormalization" else nn.LayerNorm(1)

        self.conv3 = nn.Conv1d(in_channels=params.get("filters2", 64),
                               out_channels=params.get("filters3", 32),
                               kernel_size=params.get("kernel_size3", 5),
                               stride=params.get("strides3", 3))
        self.act3 = get_activation(params.get("activation3", "elu"))
        self.norm2 = nn.BatchNorm1d(params.get("filters3", 32)) if params.get("normalization_method2", "BatchNormalization") == "BatchNormalization" else nn.LayerNorm(1)

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(None, params.get("dense_units", 16))  # will be set dynamically in forward
        self.dense_activation = get_activation(params.get("dense_activation", "sigmoid"))

        if num_classes == 2:
            self.output_layer = nn.Linear(params.get("dense_units", 16), 1)
        else:
            self.output_layer = nn.Linear(params.get("dense_units", 16), num_classes)

        self.num_classes = num_classes

    def forward(self, x):
        # input shape: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.spatial_dropout(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.act2(x)
        if isinstance(self.norm1, nn.LayerNorm):
            x = x.permute(0, 2, 1)
            x = self.norm1(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.norm1(x)
        x = self.conv3(x)
        x = self.act3(x)
        if isinstance(self.norm2, nn.LayerNorm):
            x = x.permute(0, 2, 1)
            x = self.norm2(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.norm2(x)
        x = self.flatten(x)

        # Dynamic Linear layer to handle unknown flatten size
        if not hasattr(self, '_flattened'):
            self.dense = nn.Linear(x.shape[1], self.dense.out_features).to(x.device)
            self._flattened = True

        x = self.dense(x)
        x = self.dense_activation(x)
        x = self.output_layer(x)

        if self.num_classes == 2:
            return x.squeeze(1)  # shape (B,)
        else:
            return x  # shape (B, num_classes)

def customizable_nicon_classification(input_shape, num_classes=2, params={}):
    return CustomNiconClassifier(input_shape=input_shape, num_classes=num_classes, params=params)
