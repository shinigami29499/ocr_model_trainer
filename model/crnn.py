import torch
import torch.nn as nn
import torch.nn.functional as F

from constants.config import *

# ------------------------------
# 🧱 Modular 20-layer CNN Backbone
# ------------------------------
def make_ocr_cnn(num_input_channels: int = 1) -> nn.Sequential:
    layers = []
    in_channels = num_input_channels
    out_channels_cycle = [64, 128, 256, 512]

    for i in range(20):
        out_channels = out_channels_cycle[(i // 4) % len(out_channels_cycle)]
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Reduce height, preserve width
        if (i + 1) % 4 == 0:
            if i < 8:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Downsample height & width
            else:
                layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))  # Downsample height only

        in_channels = out_channels

    return nn.Sequential(*layers)


# ------------------------------
# 🧠 CRNN Model for OCR
# ------------------------------
class CRNN(nn.Module):
    def __init__(self, num_classes: int = len(CHARSET) + 1):
        super(CRNN, self).__init__()

        # ------------------------------
        # 🧱 CNN Backbone (feature extractor)
        # ------------------------------
        self.cnn = make_ocr_cnn(num_input_channels=1)

        # ------------------------------
        # 🔄 Dynamic Projection Layer (reshape for RNN)
        # ------------------------------
        self.rnn_input_size = 512
        self.feature_proj = None  # will be initialized on first forward pass

        # ------------------------------
        # 🔁 Bidirectional RNN Layer
        # ------------------------------
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # ------------------------------
        # 🎯 Classification Layer (character prediction)
        # ------------------------------
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- CNN: Extract spatial features ---
        x = self.cnn(x)                  # (B, C, H, W)

        # --- Reshape: prepare for RNN ---
        B, C, H, W = x.size()
        x = x.permute(0, 3, 2, 1)        # (B, W, H, C)
        x = x.reshape(B, W, H * C)       # (B, W, H*C)

        # --- Project to RNN input dimension ---
        if self.feature_proj is None:
            self.feature_proj = nn.Linear(H * C, self.rnn_input_size).to(x.device)

        x = self.feature_proj(x)         # (B, W, 512)

        # --- RNN: model sequence relationships ---
        x, _ = self.rnn(x)               # (B, W, 512)
        x = self.dropout(x)

        # --- Classification ---
        x = self.fc(x)                   # (B, W, num_classes)
        x = x.permute(1, 0, 2)           # (T, B, num_classes) — required for CTC loss

        return F.log_softmax(x, dim=2)
