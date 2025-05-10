import torch
import torch.nn.functional as F
import yaml
import os
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, Subset
from waveletvae import WaveletVAE, UncertaintyLossWeighting
from data_processor import WaveformScatteringDataset
import wandb

# def cyclical_kl_beta(epoch, cycle_length=30, max_beta=0.1):
#     cycle_pos = epoch % cycle_length
#     return max_beta * min(0.1, cycle_pos / (cycle_length // 2))

def compute_kl(mu, logvar, free_bits=0.1):
    kl_per_dim = -0.5 * (1 + torch.log(logvar + 1e-6) - mu.pow(2) - logvar)
    return torch.clamp(kl_per_dim, min=free_bits).mean()

def train_and_generate(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    wandb.init(
    project="wavelet-vae",
    name="uncertainty_weighting_run",
    config=cfg  # <-- pass the loaded dict
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = WaveformScatteringDataset(
        json_file=cfg["dataset"]["json_path"]["train"],
        audio_root=cfg["dataset"]["json_path"]["audio_root"],
        sr=cfg["dataset"]["sr"],
        duration=cfg["dataset"]["duration"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        window=cfg["dataset"]["window"]
    )

    # subset = Subset(dataset, [0])
    # loader = DataLoader(subset, batch_size=1, shuffle=False)
    subset = Subset(dataset, list(range(20)))  # pick 8 samples
    loader = DataLoader(subset, batch_size=4, shuffle=False)

    model = WaveletVAE(
        input_waveform_length=cfg["model"]["input_waveform_length"],
        latent_dim=cfg["model"]["latent_dim"],
        scat_shape=tuple(cfg["model"]["scattering_shape"])
    ).to(device)
    loss_weighting = UncertaintyLossWeighting().to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_weighting.parameters()), 
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,            # Reduce LR by half
        patience=5,            # Wait 5 epochs with no improvement
        threshold=1e-2,        # Minimum change to qualify as an improvement
        min_lr=1e-6,           # Lower bound on LR
        verbose=True
    )
    num_epochs = cfg["training"].get("epochs", 100)
    ckpt_path = cfg["training"]["checkpoint_path"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    best_val_loss = float("inf")
    print(f"🔧 Training for {num_epochs} epochs on 1 sample...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        for waveform, scat, _ in loader:
            waveform, scat = waveform.to(device), scat.to(device)
            recon_wave, _, mu, logvar = model(scat)

            recon_loss = F.l1_loss(recon_wave, waveform)
            logvar = F.softplus(logvar)
            # mu = torch.tanh(mu)
            kl = compute_kl(mu, logvar)
            # beta = cyclical_kl_beta(epoch, cycle_length=30, max_beta=0.1)
            loss = loss_weighting.compute_loss(recon_loss, kl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({
                "train/recon_loss": recon_loss.item(),
                "train/kl_loss": kl.item(),
                "train/total_loss": loss.item(),
                "train/alpha": torch.exp(loss_weighting.log_sigma_recon).item(),
                "train/beta": torch.exp(loss_weighting.log_sigma_kl).item(),
                "epoch": epoch
            })


        model.eval()
        with torch.no_grad():
            for waveform, scat, _ in loader:
                waveform, scat = waveform.to(device), scat.to(device)
                recon_wave, _, mu, logvar = model(scat)
                recon_loss = F.l1_loss(recon_wave, waveform)
                logvar = F.softplus(logvar)
                # mu = torch.tanh(mu)
                kl = compute_kl(mu, logvar)
                val_loss = loss_weighting.compute_loss(recon_loss, kl)
                wandb.log({
                    "val/recon_loss": recon_loss.item(),
                    "val/kl_loss": kl.item(),
                    "val/total_loss": val_loss.item()
                })



        scheduler.step(val_loss.item())

        print(f"[Epoch {epoch:03d}] Recon: {recon_loss.item():.4f}, KL: {kl.item():.4f}, Val: {val_loss.item():.4f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), ckpt_path)
            print("✅ Best checkpoint updated.")

    print("🎯 Loading best checkpoint and generating audio...")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    test_loader = DataLoader(subset, batch_size=1, shuffle=False)

    save_dir = cfg["training"].get("save_dir", ".")
    output_subdir = os.path.join(save_dir, "generated")
    os.makedirs(output_subdir, exist_ok=True)

    sample_rate = cfg["dataset"]["sr"]

    with torch.no_grad():
        for idx, (waveform, scat, _) in enumerate(test_loader):
            waveform, scat = waveform.to(device), scat.to(device)
            recon_wave, _, _, _ = model(scat)

            recon_np = recon_wave.squeeze().cpu().numpy()
            recon_np /= np.max(np.abs(recon_np) + 1e-6)  # Normalize
            recon_tensor = torch.tensor(recon_np).unsqueeze(0)

            file_name = f"sample_{idx:03d}.wav"
            out_path = os.path.join(output_subdir, file_name)
            torchaudio.save(out_path, recon_tensor, sample_rate)

    print(f"🎧 Saved {len(subset)} reconstructed samples in: {output_subdir}")
    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train_and_generate(args.config)