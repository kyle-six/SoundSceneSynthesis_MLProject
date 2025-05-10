import torch
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import shutil


# Load the CSV containing captions and filenames
captions_file = "captions.csv"  # Path to your CSV
captions_df = pd.read_csv(captions_file)

# Folder paths for the embeddings
audio_aldm_folder = "audioldm_output/"
audio_tango_folder = "tangoFlux_output/"
audio_aldm_embeddings_folder = "audioldm_output/embeddings/panns-wavegram-logmel/"
audio_tango_embeddings_folder = "tangoFlux_output/embeddings/panns-wavegram-logmel/"
dcase_panns_folder = "dcase-panns-wavegram-logmel/"  # Folder with reference embeddings
best_outputs_folder = "best_audio/"  # Folder for saving the best audio files

# Ensure the output folder exists
if not os.path.exists(best_outputs_folder):
    os.makedirs(best_outputs_folder)

# Function to load embeddings
def load_embedding(file_path):
    return np.load(file_path)  # Assuming embeddings are stored as .npy files

# Function to get the PANN embedding for a given audio file
def get_audio_embedding(file_name, model="aldm"):
    if model == "aldm":
        file_path = os.path.join(audio_aldm_embeddings_folder, f"{file_name}.npy")
    else:
        file_path = os.path.join(audio_tango_embeddings_folder, f"{file_name}.npy")
   
    if os.path.exists(file_path):
        return load_embedding(file_path)
    else:
        print(f"Warning: {file_path} not found!")
        return None

# Function to get reference embeddings from dcase-panns
def get_reference_embedding(file_name):
    file_path = os.path.join(dcase_panns_folder, f"{file_name}.npy")
    if os.path.exists(file_path):
        return load_embedding(file_path)
    else:
        print(f"Warning: {file_path} not found!")
        return None

# Function to compute cosine similarity between two embeddings
def compute_cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.flatten() if embedding1.ndim > 1 else embedding1
    embedding2 = embedding2.flatten() if embedding2.ndim > 1 else embedding2
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Create a list to store the results (for the CSV)
results = []

# Iterate through the CSV rows and compare the two generated audios
for index, row in captions_df.iterrows():
    prompt = row['caption']
    file_name = row['file'][:7]

    # Get the audio embeddings for AudioLDM and TangoFlux
    audio_aldm_emb = get_audio_embedding(file_name, model="aldm")
    audio_tango_emb = get_audio_embedding(file_name, model="tango")

    # Get the reference embedding for the prompt (DCASE PANN)
    ref_emb = get_reference_embedding(file_name)

    if audio_aldm_emb is not None and audio_tango_emb is not None and ref_emb is not None:
        # Compute cosine similarity between the generated audio and the reference embedding
        sim_aldm = compute_cosine_similarity(audio_aldm_emb, ref_emb)
        sim_tango = compute_cosine_similarity(audio_tango_emb, ref_emb)

        # Print similarities
        print(f"AudioLDM similarity to reference for {file_name}: {sim_aldm:.4f}")
        print(f"TangoFlux similarity to reference for {file_name}: {sim_tango:.4f}")

        # Save the result to the results list
        results.append({
            "file_name": file_name,
            "caption": prompt,
            "sim_aldm": sim_aldm,
            "sim_tango": sim_tango,
            "best_audio": "AudioLDM" if sim_aldm > sim_tango else "TangoFlux"
        })

        # Select the better audio based on similarity
        if sim_aldm > sim_tango:
            # If AudioLDM is better, select it
            selected_audio_path = os.path.join(audio_aldm_folder, f"{file_name}.wav")
            shutil.copy(selected_audio_path, os.path.join(best_outputs_folder, f"{file_name}_best.wav"))
            print(f"Selected AudioLDM generated audio as the best for aldm_{file_name}")
        else:
            # If TangoFlux is better, select it
            selected_audio_path = os.path.join(audio_tango_folder, f"{file_name}.wav")
            shutil.copy(selected_audio_path, os.path.join(best_outputs_folder, f"{file_name}_best.wav"))
            print(f"Selected TangoFlux generated audio as the best for tangoFlux_{file_name}")
    else:
        print(f"Skipping {file_name} due to missing embeddings.")

# Save the results to a CSV file
output_csv = "audio_similarity_results.csv"
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)

print(f"Best audio saved to {best_outputs_folder}")
print(f"Similarity results saved to {output_csv}")
