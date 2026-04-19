# Deep Convolutional AutoEncoder-based Lossy Image Compression 2018
import torch
import torch.nn as nn

from .base import BaseAutoencoder
import torch.nn.functional as F

import constriction
import numpy as np
import dahuffman


class DownBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2

        def downsample_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, stride=2, padding=padding),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            downsample_block(in_channels, out_channels),
            downsample_block(out_channels, out_channels),
            downsample_block(out_channels, out_channels),
            downsample_block(out_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class UpBranch(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2

        def upsample_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, padding=padding),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(out_c, out_c * 4, kernel_size=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            upsample_block(in_channels, out_channels),
            upsample_block(out_channels, out_channels),
            upsample_block(out_channels, out_channels),
            upsample_block(out_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.row1 = DownBranch(3, 42, kernel_size=3)
        self.row2 = DownBranch(3, 42, kernel_size=5)
        self.row3 = DownBranch(3, 44, kernel_size=7)

        self.pca_rotation = nn.Conv2d(128, 128, kernel_size=1, bias=False)

    def forward(self, x):
        out1 = self.row1(x)
        out2 = self.row2(x)
        out3 = self.row3(x)

        x_concat = torch.cat([out1, out2, out3], dim=1)
        return self.pca_rotation(x_concat)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inverse_pca = nn.Conv2d(128, 128, kernel_size=1, bias=False)

        self.row1 = UpBranch(42, 42, kernel_size=3)
        self.row2 = UpBranch(42, 42, kernel_size=5)
        self.row3 = UpBranch(44, 44, kernel_size=7)

        self.final_merge = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.inverse_pca(x)

        x1, x2, x3 = torch.split(x, [42, 42, 44], dim=1)

        out1 = self.row1(x1)
        out2 = self.row2(x2)
        out3 = self.row3(x3)

        x_concat = torch.cat([out1, out2, out3], dim=1)
        return self.final_merge(x_concat)


class DCAL_2018(BaseAutoencoder):
    def __init__(self, learning_rate=1e-3):
        super().__init__(learning_rate=learning_rate)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantization_bits = 12 # lower -> more compression
        self.rate_coeffitient = 0.5  # higher -> more compression
        assert self.rate_coeffitient >= 0.0

    def pass_to_encoders(self, x):
        encoded = self.encoder(x)
        return encoded

    def pass_to_decoders(self, x):
        decoded = self.decoder(x)
        return decoded

    def entropy_coder(self, x):
        symbols = x.cpu().numpy().astype(np.int32).flatten()
        # tile to match batch dimension in symbols
        means = np.tile(self.z_means.cpu().numpy().flatten(), x.shape[0]).astype(np.float64)
        stds = np.tile((self.z_stds + 1e-8).cpu().numpy().flatten(), x.shape[0]).astype(np.float64)

        B = self.quantization_bits
        model_family = constriction.stream.model.QuantizedGaussian(-2**(B-1), 2**(B-1)-1)
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(symbols, model_family, means, stds)
        compressed =  coder.get_compressed()

        return compressed

    def entropy_decoder(self, x, original_shape):
        # tile to match batch size
        batch_size = original_shape[0]
        means = np.tile(self.z_means.cpu().numpy().flatten(), batch_size).astype(np.float64)
        stds = np.tile((self.z_stds + 1e-8).cpu().numpy().flatten(), batch_size).astype(np.float64)

        B = self.quantization_bits
        model_family = constriction.stream.model.QuantizedGaussian(-2**(B-1), 2**(B-1)-1)
        decoder = constriction.stream.stack.AnsCoder(x)
        symbols = decoder.decode(model_family, means, stds)
        return torch.tensor(symbols, dtype=torch.int32).reshape(original_shape).to(next(self.parameters()).device)

    def pca_rotation(self, x):
        # TODO proper PCA
        y = (x - self.z_means) / (self.z_stds+ 1e-8)
        return y

    def pca_inverse(self, x):
        # TODO proper PCA
        return x * self.z_stds + self.z_means

    def quantizer(self, x):
        B = self.quantization_bits
        x_quantized = torch.round(2**(B-1) * x).clamp(-2**(B-1), 2**(B-1)-1).to(torch.int32)
        return x_quantized

    def dequantizer(self, x):
        B = self.quantization_bits
        x_reconstructed = (x / 2**(B-1))
        return x_reconstructed

    def compute_priors(self, all_latents):
        z_means = all_latents.mean(dim=0)
        z_stds = all_latents.std(dim=0)
        self.register_buffer('z_means', z_means)
        self.register_buffer('z_stds', z_stds)

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.pass_to_encoders(x)
        #z = self.encoder(x)

        # add uniform noise to simulate quantization
        noise = torch.zeros_like(z).uniform_(-(1.0/1024.0), 1.0/1024.0)

        # x_hat = self.decoder(z+noise)
        x_hat = self.pass_to_decoders(z+noise)

        loss = F.mse_loss(x_hat, x) + self.rate_coeffitient*torch.mean(z ** 2)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        # z = self.encoder(x)
        z = self.pass_to_encoders(x)
        # x_hat = self.decoder(z)
        x_hat = self.pass_to_decoders(z)

        loss = F.mse_loss(x_hat, x) + self.rate_coeffitient*torch.mean(z ** 2)
        self.log("val_loss", loss, prog_bar=True)

    def forward(self, x):
        x_hat, _= self.forward_get_latent(x)
        return x_hat

    def forward_get_latent(self, x):
        # z = self.encoder(x)
        z = self.pass_to_encoders(x)
        z_rot = self.pca_rotation(z)
        z_q = self.quantizer(z_rot)

        USE_FANCY_COMPRESSION = True
        if USE_FANCY_COMPRESSION:
            z_compressed = self.entropy_coder(z_q)
            original_shape = z_q.shape
            z_decompressed = self.entropy_decoder(z_compressed, original_shape)
            #
            z_compressed_data = z_compressed.tobytes()
            #
        else:
            symbols = z_q.cpu().numpy().astype(np.int32).flatten()
            codec = dahuffman.HuffmanCodec.from_data(symbols)
            z_compressed = codec.encode(symbols)
            z_decompressed = torch.tensor(codec.decode(z_compressed), dtype=torch.int32).reshape(z_q.shape).to(next(self.parameters()).device)
            z_compressed_data = z_compressed

        z_deq = self.dequantizer(z_decompressed)
        z_inv_rot = self.pca_inverse(z_deq)
        # x_hat = self.decoder(z_inv_rot)
        x_hat = self.pass_to_decoders(z_inv_rot)

        return (x_hat, z_compressed_data)