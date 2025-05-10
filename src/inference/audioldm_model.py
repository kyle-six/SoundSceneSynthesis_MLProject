from .model_interface import ModelInterface
#from audioldm import text_to_audio, build_model
from diffusers import AudioLDMPipeline

import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

class AudioldmModel(ModelInterface):
    DEFAULT_REPO_ID = "cvssp/audioldm-s-full-v2"
    
    INFERENCE_STEPS = 10
    
    #GEN_DURATION_SEC = 5.0
    #TARGET_DURATION_SEC = 4.0
    SAMPLE_RATE = 32000
    HOP_DURATION_SEC = 0.1

    def __init__(self):
        super().__init__()
        self.model = None

    def load(self, path: str = ""):
        if self.model is not None:
            print("Warning: AudioLDM model already loaded.")
            return
        
        if path == "" or not Path(path).exists():
            print("Checkpoint file not found, loading pre-trained model...")
            self.model = AudioLDMPipeline.from_pretrained(self.DEFAULT_REPO_ID, torch_dtype=torch.float16)
        else:
            try:
                self.model = AudioLDMPipeline.from_pretrained(path, torch_dtype=torch.float16)
            except:
                raise ValueError("Incorrect checkpoint path to load AudioLDM model.")  
            
        # Move to correct GPU/CPU
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        print("AudioLDM model loaded.")
        
    def save(self, path: str = ""):
        if path == "":
            print("Error: A save path was not specified!")

    def train(self):
        pass

    def infer(self, prompt: str, duration: float):
        if self.model is None:
            print("Error: Model not loaded.")
            return None

        waveform = self.model(prompt, num_inference_steps=self.INFERENCE_STEPS, audio_length_in_s=duration + 1.0).audios[0]

        # Extract loudest 4 seconds of audio
        cropped_waveform = self.find_loudest_segment(waveform, sr=self.SAMPLE_RATE, segment_length=int(duration), hop_length_sec=self.HOP_DURATION_SEC)
        cropped_waveform = torch.tensor(cropped_waveform, dtype=torch.float32).unsqueeze(0)
        
        return cropped_waveform
    
    
    @staticmethod
    def find_loudest_segment(audio: np.ndarray, sr: int, segment_length: int = 4, hop_length_sec: float = 1.0) -> np.ndarray:
        hop_length_samples = int(sr * hop_length_sec)
        frame_length_samples = int(sr * segment_length)

        if len(audio) < frame_length_samples:
            return np.pad(audio, (0, frame_length_samples - len(audio)), mode='constant')

        rms_values = librosa.feature.rms(
            y=audio,
            frame_length=frame_length_samples,
            hop_length=hop_length_samples,
            center=False
        ).squeeze()

        max_rms_index = np.argmax(rms_values)
        start_sample = max_rms_index * hop_length_samples
        end_sample = start_sample + frame_length_samples

        return audio[start_sample:end_sample]

