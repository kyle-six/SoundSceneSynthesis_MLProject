import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyLossWeighting(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_recon = nn.Parameter(torch.tensor(0.0))  # log(σ_recon)
        self.log_sigma_kl = nn.Parameter(torch.tensor(0.0))     # log(σ_kl)

    def compute_loss(self, recon_loss, kl_loss):
        sigma_recon_sq = torch.exp(self.log_sigma_recon) ** 2
        sigma_kl_sq = torch.exp(self.log_sigma_kl) ** 2

        weighted_recon = (recon_loss / (2 * sigma_recon_sq)) + self.log_sigma_recon
        weighted_kl = (kl_loss / (2 * sigma_kl_sq)) + self.log_sigma_kl

        return weighted_recon + weighted_kl


class WaveletVAE(nn.Module):
    def __init__(self, input_waveform_length=320000, latent_dim=128, scat_shape=(217, 64, 313)):
        super().__init__()
        self.input_len = input_waveform_length
        self.scat_shape = scat_shape  # Full 217 channels
        C, H, W = scat_shape

        # === Encoder ===
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            dummy_out = self.encoder(dummy)
            B, C_out, H_out, W_out = dummy_out.shape
            self.flat_dim = C_out * H_out * W_out
            self.H_out = H_out
            self.W_out = W_out

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # === Decoder 2D ===
        self.decoder_2d = nn.Sequential(
            nn.Unflatten(1, (512, self.H_out, self.W_out)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, C, 4, stride=2, padding=1),
            nn.GELU()
        )

        # === Decoder 1D for waveform ===
        self.final_decoder_1d = nn.Sequential(
            nn.Conv1d(C, 256, 7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.ConvTranspose1d(256, 128, 9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, 9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, 9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, 9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.ConvTranspose1d(16, 8, 9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.ConvTranspose1d(8, 1, 9, stride=2, padding=4, output_padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-8, max=4)  # Tighter clamp
        mu = F.layer_norm(mu, mu.shape)              # Normalize latent mean
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, scat_input):
        scat_input = scat_input.squeeze(1)  # [B, 1, 217, H, W] → [B, 217, H, W]
        mu, logvar = self.encode(scat_input)
        z = self.reparameterize(mu, logvar)
        h = self.fc_decode(z)
        scat_recon = self.decoder_2d(h)
        scat_1d = scat_recon.view(scat_recon.size(0), scat_recon.size(1), -1)
        waveform = self.final_decoder_1d(scat_1d)
        waveform = waveform[..., :self.input_len]
        return waveform, scat_recon, mu, logvar
