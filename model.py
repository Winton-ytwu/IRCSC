import torch
import torch.nn as nn
from utils import awgn_channel, rayleigh_channel

class JSCC_Classifier(nn.Module):
    def __init__(self, k):
        super(JSCC_Classifier, self).__init__()
        self.k = k
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, 2 * self.k, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * self.k, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, snr_db=10, channel_type="awgn", return_features=False):
        
        encoded_interleaved = self.encoder(x)
        
        z_tilde = torch.complex(encoded_interleaved[:, 0::2], encoded_interleaved[:, 1::2])
        z = z_tilde 
        
        if channel_type == "awgn":
            z_noisy = awgn_channel(z, snr_db)
        elif channel_type == "rayleigh":
            z_noisy = rayleigh_channel(z, snr_db)
        else:
            raise ValueError("Unsupported channel type.")
            
        x_noisy = torch.empty_like(encoded_interleaved)
        x_noisy[:, 0::2] = torch.real(z_noisy)
        x_noisy[:, 1::2] = torch.imag(z_noisy)
        
        logits = self.decoder(x_noisy)
        if return_features:
            return logits, encoded_interleaved
        else:
            return logits