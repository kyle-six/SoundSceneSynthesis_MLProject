import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from data_processor import WaveformScatteringDataset
from waveletvae import WaveletVAE

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.autograd.set_detect_anomaly(True)  # Optional: for debugging exploding gradients

    # === Dataset ===
    train_set = WaveformScatteringDataset(
        json_file=cfg["dataset"]["json_path"]["train"],
        audio_root=cfg["dataset"]["json_path"]["audio_root"],
        sr=cfg["dataset"]["sr"],
        duration=cfg["dataset"]["duration"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        window=cfg["dataset"]["window"]
    )

    val_set = WaveformScatteringDataset(
        json_file=cfg["dataset"]["json_path"]["val"],
        audio_root=cfg["dataset"]["json_path"]["audio_root"],
        sr=cfg["dataset"]["sr"],
        duration=cfg["dataset"]["duration"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        window=cfg["dataset"]["window"]
    )

    train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=cfg["training"]["batch_size"], num_workers=4)

    # === Model ===
    model = WaveletVAE(
        input_waveform_length=cfg["model"]["input_waveform_length"],
        latent_dim=cfg["model"]["latent_dim"],
        scat_shape=tuple(cfg["model"]["scattering_shape"])  # (217, H, W)
    ).to(device)

    learning_rate = float(cfg["training"]["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_val = float("inf")

    print("🔧 Training started...")
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        train_recon, train_kl = 0, 0

        for waveform, scat, _ in train_loader:
            waveform = waveform.to(device)
            scat = scat.to(device)  # ✅ No unsqueeze

            # Optional: Debug check
            print("waveform nan:", torch.isnan(waveform).any(), "scat nan:", torch.isnan(scat).any())

            recon_wave, _, mu, logvar = model(scat)

            logvar = torch.clamp(logvar, -10.0, 10.0)
            recon_loss = F.mse_loss(recon_wave + 1e-8, waveform + 1e-8)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_recon += recon_loss.item()
            train_kl += kl.item()

        # === Validation ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for waveform, scat, _ in val_loader:
                waveform = waveform.to(device)
                scat = scat.to(device)

                recon_wave, _, mu, logvar = model(scat)
                logvar = torch.clamp(logvar, -10.0, 10.0)
                loss = F.mse_loss(recon_wave + 1e-8, waveform + 1e-8) + \
                       (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
                val_loss += loss.item()

        train_recon /= len(train_loader)
        train_kl /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"[Epoch {epoch+1:03d}] Recon: {train_recon:.4f} | KL: {train_kl:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg["training"]["checkpoint_path"])
            print("✅ Saved best checkpoint.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
