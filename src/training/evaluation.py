import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from data_processor import WaveformScatteringDataset
from waveletvae import WaveletVAE
import torchaudio
import os

@torch.no_grad()
def evaluate(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Load test set ===
    test_set = WaveformScatteringDataset(
        json_file=cfg["dataset"]["json_path"]["test"],
        sr=cfg["dataset"]["sr"],
        duration=cfg["dataset"]["duration"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"]
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # === Load model ===
    model = WaveletVAE(
        input_waveform_length=cfg["model"]["input_waveform_length"],
        latent_dim=cfg["model"]["latent_dim"],
        scat_shape=tuple(cfg["model"]["scattering_shape"])
    ).to(device)

    model.load_state_dict(torch.load(cfg["training"]["checkpoint_path"], map_location=device))
    model.eval()

    recon_losses = []
    kl_losses = []
    save_dir = os.path.join("outputs", "reconstructions")
    os.makedirs(save_dir, exist_ok=True)

    print("🔍 Evaluating on test set...")
    for idx, (waveform, scat, caption) in enumerate(test_loader):
        waveform, scat = waveform.to(device), scat.to(device)

        recon_wave, _, mu, logvar = model(scat)
        recon_loss = F.mse_loss(recon_wave, waveform)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        recon_losses.append(recon_loss.item())
        kl_losses.append(kl.item())

        if idx < 10:  # Save first 10 reconstructions
            torchaudio.save(
                os.path.join(save_dir, f"recon_{idx:03d}.wav"),
                recon_wave.cpu().squeeze(0),
                cfg["dataset"]["sr"]
            )
            torchaudio.save(
                os.path.join(save_dir, f"input_{idx:03d}.wav"),
                waveform.cpu().squeeze(0),
                cfg["dataset"]["sr"]
            )
            with open(os.path.join(save_dir, f"caption_{idx:03d}.txt"), "w") as f:
                f.write(caption[0])

    print(f"✅ Evaluation complete. Test MSE: {sum(recon_losses)/len(recon_losses):.4f}, KL: {sum(kl_losses)/len(kl_losses):.4f}")
    print(f"🎧 Reconstructed examples saved to: {save_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.config)
