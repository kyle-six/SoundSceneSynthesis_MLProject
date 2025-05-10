import json
import torch
import torchaudio
from torch.utils.data import Dataset
from kymatio.torch import Scattering2D
from pathlib import Path

class WaveformScatteringDataset(Dataset):
    def __init__(self, json_file, audio_root, sr=32000, duration=10, n_fft=1024, hop_length=256, window="hann"):
        with open(json_file, 'r') as f:
            self.entries = json.load(f)["data"]

        self.audio_root = Path(audio_root)
        self.sr = sr

        self.samples = sr * duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.scattering = Scattering2D(J=3, shape=(513, 1251))  # Adjust shape if needed
        if window == "hann":
            self.window = torch.hann_window(n_fft)
        elif window == "hamming":
            self.window = torch.hamming_window(n_fft)
        elif window == "blackman":
            self.window = torch.blackman_window(n_fft)
        elif window == "rectangular" or window is None:
            self.window = None
        else:
            raise ValueError(f"Unsupported window type: {window}")
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        relative_path = entry["wav"]
        audio_path = self.audio_root / relative_path
        caption = entry["caption"]

        waveform, sr = torchaudio.load(str(audio_path))
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        waveform = waveform.mean(0, keepdim=True)
        waveform = torch.nn.functional.pad(waveform, (0, max(0, self.samples - waveform.shape[-1])))
        waveform = waveform[:, :self.samples]

        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length,window=self.window, return_complex=True)
        log_mag = torch.log1p(stft.abs()).unsqueeze(0).contiguous()
        scat = self.scattering(log_mag).squeeze(1)

        return waveform, scat, caption
