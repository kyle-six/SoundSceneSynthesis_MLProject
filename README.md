# SoundSceneSynthesis_MLProject
Authors:
- Chhatrapathi Sivaji Lakkimsetty
- Josh Manogaran
- Kyle Six
- Kahlia Gronthos

This pipeline seeks to improve upon the submissions to the DCASE Challenge 2024 under Task 7: Sound Scene Synthesis. 
Following from the state-of-the-art approach which uses AudioLDM in combination with Tango2, we have chosen to adapt and implement the
newer "TangoFlux" model to achieve higher audio-perception scores in the task. 
    Our improved approach follows:
    1) Load AudioLDM and TangoFlux
    2) Run inference on both models to get 2 potential waveforms
    3) Compute audio embeddings of both waveforms
    4) Selectively replace the waveform using the cosine similarity of the embeddings
    5) Evaluate the FAD score between our selected audio and the DCASE baseline
We also modified the newer TangoFlux model to utilize Wavelet Scattering to improve further. The beginning results of training this new model are also stored here.

    Note, due to incompatibilities between the required components of our approach (AudioLDM (as in DCASE), TangoFlux, and Fad Toolkit)
we were forced to use separate python environments for each stage of the pipeline. This limited our integration of components and relies on
shell commands and sharing files via local directories. However, the final result indicates that our approach improves upon the SOTA


# Setup
1) git clone https://github.com/kyle-six/SoundSceneSynthesis_MLProject.git
- Clone this repo to wherever you like
2) cd SoundSceneSynthesis_MLProject
- Change directory to the root of the repository
3) python -m pip install requirements.txt
- install depedencies for the pipeline script ONLY

# AudioLDM 

Download the small AudioLDM model fine-tuned with AudioCaps and MusicCaps audio-text pairs from [Hugging Face](https://huggingface.co/circulus/AudioLDM/blob/main/audioldm-s-text-ft.ckpt). Place the 'audioldm-s-text-ft.ckpt' file in the project root directory before running the model.

# TangoFlux

Automatically downloaded with dependencies in the src/inference/requirements.txt 

# Useage Instructions
python pipeline.py
- This runs ALL of the below modules from start -> finish, navigating incompatible dependencies by creating multiple envs
- NOTE: this script expects to be run on an NT-based shell (e.g. Windows)!

python src/inference/main.py --dataset_folder "{PATH_TO_DCASE_DEVELOPMENT_DIR}" --output_folder "{PATH_TO_DESIRED_OUTPUT_DIR}"
- Can be run manually if you are using a virtual environment with src/inference/requirements.txt installed

