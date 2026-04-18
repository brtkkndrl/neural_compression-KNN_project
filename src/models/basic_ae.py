import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAutoencoder


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x

class BasicAE(BaseAutoencoder):
    def __init__(self, learning_rate=1e-3):
        super().__init__(learning_rate=learning_rate)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.name = "BasicAE"
        self.quantization_bits = 12 # lower -> more compression
        self.rate_coeffitient = 0.2  # higher -> more compression

    def entropy_coder(self, x):
        # TODO
        return x
    
    def entropy_decoder(self, x):
        # TODO
        return x

    def pca_rotation(self, x):
        # TODO proper PCA
        return (x - self.z_means) / self.z_stds

    def pca_inverse(self, x):
        # TODO proper PCA
        return x * self.z_stds + self.z_means

    def quantizer(self, x):
        B = self.quantization_bits
        x_quantized = torch.round(2**(B-1) * x).clamp(-2**(B-1), 2**(B-1)-1)
        return x_quantized

    def dequantizer(self, x):
        B = self.quantization_bits
        x_reconstructed = (x / 2**(B-1))
        return x_reconstructed

    def compute_priors(self, all_latents):
        self.z_means = all_latents.mean(dim=0)
        self.z_stds = all_latents.std(dim=0)

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)

        # add uniform noise to simulate quantization
        noise = torch.zeros_like(z).uniform_(-(1.0/1024.0), 1.0/1024.0)

        x_hat = self.decoder(z+noise)

        loss = F.mse_loss(x_hat, x) + self.rate_coeffitient*torch.mean(z ** 2)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x) + self.rate_coeffitient*torch.mean(z ** 2)
        self.log("val_loss", loss, prog_bar=True)

    def forward(self, x):
        x_hat, _= self.forward_get_latent(x)
        return x_hat

    def forward_get_latent(self, x):
        z = self.encoder(x)
        z_rot = self.pca_rotation(z)
        z_q = self.quantizer(z_rot)
        z_compressed = self.entropy_coder(z_q)

        z_decompressed = self.entropy_decoder(z_compressed)
        z_deq = self.dequantizer(z_decompressed)
        z_inv_rot = self.pca_inverse(z_deq)
        x_hat = self.decoder(z_inv_rot)
        return (x_hat, z_compressed)
