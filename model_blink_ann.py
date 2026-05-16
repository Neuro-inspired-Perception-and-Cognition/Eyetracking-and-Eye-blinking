import torch
import torch.nn as nn


class ANN_Model(nn.Module):
    def __init__(self, gru_hidden_size=85, num_classes=4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Temporal Aggregation
        self.cnn_output_dim = 128 * 4 * 4

        self.gru = nn.GRU(input_size=self.cnn_output_dim, hidden_size=gru_hidden_size, batch_first=True)

        self.fc = nn.Linear(gru_hidden_size, num_classes)

    def forward(self, x, hx=None):
        b, t, c, h, w = x.shape
        c_in = x.view(b * t, c, h, w)
        features = self.cnn(c_in)
        features = features.view(b, t, -1)
        gru_out, hx = self.gru(features, hx)
        return self.fc(gru_out), hx


if __name__ == "__main__":
    model = ANN_Model(gru_hidden_size=85)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ANN Parameter Count: {params:,}")