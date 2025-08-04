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
    
def compute_output_length(L_in, kernel_size, stride):
        return (L_in - kernel_size) // stride + 1

class CustomNiconClassifier(nn.Module):
    def __init__(self, input_shape, num_classes=2, params={}):
        super().__init__()
        in_channels = input_shape[-1]
        input_length = input_shape[-2]
        # Retrieve hyperparams related to convolutions
        k1 = params.get("kernel_size1", 15)
        s1 = params.get("strides1", 5)
        k2 = params.get("kernel_size2", 21)
        s2 = params.get("strides2", 3)
        k3 = params.get("kernel_size3", 5)
        s3 = params.get("strides3", 3)

        # Verification of the sizes
        l1 = compute_output_length(input_length, k1, s1)
        if l1 <= 0:
            raise ValueError(f"Invalid kernel_size1={k1} for input_length={input_length}")
        l2 = compute_output_length(l1, k2, s2)
        if l2 <= 0:
            raise ValueError(f"Invalid kernel_size2={k2} after conv1 output length={l1}")
        l3 = compute_output_length(l2, k3, s3)
        if l3 <= 0:
            raise ValueError(f"Invalid kernel_size3={k3} after conv2 output length={l2}")

        # Construction of the layers
        self.spatial_dropout = SpatialDropout1D(params.get("spatial_dropout", 0.08))

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=params.get("filters1", 8),
                               kernel_size=k1,
                               stride=s1)
        self.act1 = get_activation(params.get("activation1", "selu"))
        self.dropout1 = nn.Dropout(params.get("dropout_rate", 0.2))

        self.conv2 = nn.Conv1d(in_channels=params.get("filters1", 8),
                               out_channels=params.get("filters2", 64),
                               kernel_size=k2,
                               stride=s2)
        self.act2 = get_activation(params.get("activation2", "relu"))
        self.norm1 = (
            nn.BatchNorm1d(params.get("filters2", 64))
            if params.get("normalization_method1", "BatchNormalization") == "BatchNormalization"
            else nn.LayerNorm(params.get("filters2", 64))
        )

        self.conv3 = nn.Conv1d(in_channels=params.get("filters2", 64),
                               out_channels=params.get("filters3", 32),
                               kernel_size=k3,
                               stride=s3)
        self.act3 = get_activation(params.get("activation3", "elu"))
        self.norm2 = (
            nn.BatchNorm1d(params.get("filters3", 32))
            if params.get("normalization_method2", "BatchNormalization") == "BatchNormalization"
            else nn.LayerNorm(params.get("filters3", 32))
        )

        self.flatten = nn.Flatten()
        self.dense = nn.LazyLinear(params.get("dense_units", 16))
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

        if x.shape[-1] < self.conv1.kernel_size[0]:
            raise RuntimeError(f"Input length {x.shape[-1]} is smaller than conv1 kernel size {self.conv1.kernel_size[0]}")
        x = self.conv1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        if x.shape[-1] < self.conv2.kernel_size[0]:
            raise RuntimeError(f"Input length {x.shape[-1]} is smaller than conv2 kernel size {self.conv2.kernel_size[0]}")
        x = self.conv2(x)
        x = self.act2(x)
        if isinstance(self.norm1, nn.LayerNorm):
            x = x.permute(0, 2, 1)
            x = self.norm1(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.norm1(x)
        
        if x.shape[-1] < self.conv3.kernel_size[0]:
            raise RuntimeError(f"Input length {x.shape[-1]} is smaller than conv3 kernel size {self.conv3.kernel_size[0]}")
        x = self.conv3(x)
        x = self.act3(x)
        if isinstance(self.norm2, nn.LayerNorm):
            x = x.permute(0, 2, 1)
            x = self.norm2(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.norm2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dense_activation(x)
        x = self.output_layer(x)
        if self.num_classes == 2:
            return x.squeeze(1)  # shape (B,)
        else:
            return x  # shape (B, num_classes)

def customizable_nicon_classification(input_shape, num_classes=2, params={}):
    return CustomNiconClassifier(input_shape=input_shape, num_classes=num_classes, params=params)