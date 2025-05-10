'''
DCASE CHALLENGE 2024, Task 7: Sound Scene Synthesis
Machine Listening Project

Description:
    This pipeline seeks to improve upon the submissions to the DCASE Challenge 2024 under Task 7: Sound Scene Synthesis. 
Following from the state-of-the-art approach using AudioLDM in combination with Tango2, we have chosen to adapt and implement the
newer "TangoFlux" model to achieve higher audio-perception scores in the task. 
    Our improved approach follows:
    1) Load AudioLDM and TangoFlux
    2) Run inference on both models to get 2 potential waveforms
    3) Compute audio embeddings of both waveforms
    4) Selectively replace the waveform using the cosine similarity of the embeddings
    5) Evaluate the FAD score between our selected audio and the DCASE baseline
    
    Note, due to incompatibilities between the required components of our approach (AudioLDM (as in DCASE), TangoFlux, and Fad Toolkit)
we were forced to use separate python environments for each stage of the pipeline. This limited our integration of components and relies on
shell commands and sharing files via local directories. However, the final result indicates that our approach improves upon the SOTA

Authors:
    Chhatrapathi Sivaji Lakkimsetty
    Josh Manogaran
    Kyle Six
    Kahlia Gronthos
'''

import os, subprocess, time
import numpy as np
from pathlib import Path

print("#################### Sound Scene Synthesis ##################")
print(f"Operating System: {os.name}")
if os.name != "nt":
    print("Non-windows environment! This pipeline uses nt-based shell commands")
    
PROJECT_ROOT = Path(__file__).parent

DATASET_ROOT = PROJECT_ROOT.joinpath("dataset/dcase2024_task7_development")

OUTPUT_ROOT = PROJECT_ROOT.joinpath("outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Create new output folder for this run
current_time = time.strftime("%Y%m%d-%H%M%S")
output_folder = OUTPUT_ROOT.joinpath(f"{current_time}")
output_folder.mkdir(parents=True, exist_ok=False)

def custom_env_commands(env_name: str) -> str:
    return f"conda create -y --name mlproject-{env_name} python=3.9 && conda activate mlproject-{env_name} && python -m pip install -r src/{env_name}/requirements.txt &&"

############### Inference ###############
'''
Computes the waveforms using the appropraite versions of AudioLDM and TangoFlux
'''
#os.system("")
command_result = os.system(
    custom_env_commands("inference") +
    f"python src/inference/main.py --dataset_folder={DATASET_ROOT} --output_folder={output_folder}"
    )
print(command_result)

############### Embeddings & Cosine Similarity ###############
'''
Computes the pann-wavegram-logmel embeddings from the waveforms. Alos computes the cosine similarity between both models' embeddings and selects the higher as the output
'''
# Generate embeddings
command_result = os.system(
    custom_env_commands("embeddings") +
    f"fadtk.embeds panns-wavegram-logmel -d {output_folder.joinpath("waveforms")}"
    )
print(command_result)
# Compute cosine similarity
command_result = os.system(
    "conda activate mlproject-embeddings &&" +
    f"python src/embeddings/main.py --dataset_folder={DATASET_ROOT} --output_folder={output_folder} --embeddings_folder={output_folder.joinpath("embeddings/panns-wavegram-logmel")}"
    )
print(command_result)

############### FAD Evaluation ###############
'''
Computes the Frechet Audio Distance between our selected audio and the given reference audio in DCASE 2024 Task7
'''
command_result = os.system(
    custom_env_commands("fad") +
    f"python src/fad/main.py --dataset_folder={DATASET_ROOT} --output_folder={output_folder} --selected_audio_folder={output_folder.joinpath("selected_audio")}"
    )
print(command_result)

fad_score = np.load(output_folder.joinpath("FINAL_FAD_SCORE.npy"))
print(f"#####################################\n -> Final FAD Score: {fad_score:.3f}")