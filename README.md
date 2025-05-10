# SoundSceneSynthesis_MLProject

# Setup
1) Clone this repo to wherever you like
2) cd SoundSceneSynthesis_MLProject
- Change directory to the root of the repository
3) python -m pip install requirements.txt

# AudioLDM 

Download the small AudioLDM model fine-tuned with AudioCaps and MusicCaps audio-text pairs from [Hugging Face](https://huggingface.co/circulus/AudioLDM/blob/main/audioldm-s-text-ft.ckpt). Place the 'audioldm-s-text-ft.ckpt' file in the project root directory before running the model.

# TangoFlux

Automatically downloaded with dependencies in the src/inference/requirements.txt 

# Useage Instructions
python pipeline.py
- This runs ALL of the below modules from start -> finish, navigating incompatible dependencies by creating multiple envs
- NOTE: this script expects to be run on an NT-based shell (e.g. Windows)!

python src/inference/main.py --dataset_folder "{PATH_TO_DCASE_DEVELOPMENT_DIR}" --output_folder "{PATH_TO_DESIRED_OUTPUT_DIR}"

python src/embeddings/main.py --dataset_folder "{PATH_TO_DCASE_DEVELOPMENT_DIR}" --output_folder "{PATH_TO_DESIRED_OUTPUT_DIR}" --

python src/fad/main.py --dataset_folder "{PATH_TO_DCASE_DEVELOPMENT_DIR}" --output_folder "{PATH_TO_DESIRED_OUTPUT_DIR}"

