import torch
import torchaudio
import yaml
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from waveletvae import WaveletVAE
from data_processor import WaveformScatteringDataset

@torch.no_grad()
def generate_from_checkpoint(config_path):
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset & dataloader
    dataset = WaveformScatteringDataset(
        json_file=cfg["dataset"]["json_path"]["train"],
        audio_root=cfg["dataset"]["json_path"]["audio_root"],
        sr=cfg["dataset"]["sr"],
        duration=cfg["dataset"]["duration"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        window=cfg["dataset"]["window"]
    )
    subset = Subset(dataset, list(range(1)))  # just generate for 1 sample
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    # Load model
    model = WaveletVAE(
        input_waveform_length=cfg["model"]["input_waveform_length"],
        latent_dim=cfg["model"]["latent_dim"],
        scat_shape=tuple(cfg["model"]["scattering_shape"])
    ).to(device)
    model.load_state_dict(torch.load(cfg["training"]["checkpoint_path"], map_location=device))
    model.eval()

    # Inference
    for waveform, scat, _ in loader:
        scat = scat.to(device)
        recon_wave, _, _, _ = model(scat)

    # Post-process
    recon_np = recon_wave.squeeze().cpu().numpy()
    recon_np /= np.max(np.abs(recon_np) + 1e-6)
    recon_tensor = torch.tensor(recon_np)

    if recon_tensor.ndim == 1:
        recon_tensor = recon_tensor.unsqueeze(0)  # [T] → [1, T]
    elif recon_tensor.ndim == 3:
        recon_tensor = recon_tensor.squeeze(0)

    # Save
    save_dir = cfg["training"].get("save_dir", ".")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "reconstructed_sample.wav")
    torchaudio.save(out_path, recon_tensor, cfg["dataset"]["sr"])
    print(f"✅ Reconstructed audio saved at: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    generate_from_checkpoint(args.config)
