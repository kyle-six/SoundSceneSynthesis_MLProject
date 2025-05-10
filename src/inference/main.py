import torchaudio, csv, argparse, time
from pathlib import Path
from audioldm_model import AudioldmModel
from tangoflux_model import TangoFluxModel


SAMPLE_RATE = 32000

pann_embeddings = "insert path here..."

def parse_csv(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row['file']
            sentence = row['caption']
            data.append((filename, sentence))
    return data

if __name__ == "__main__":
    # Do cmd arg parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="", help=" Directory containing the .wav, .csv, and .npy emdeddings from the DCASE development dataset")
    parser.add_argument("--output_folder", type=str, default="", help=" Directory where generated audio should go")
    
    parser.add_argument("--audioldm_ckpt", type=str, default="", help="Specify path to AudioLDM checkpoint and configs (.ckpt, .json, etc..)")
    parser.add_argument("--tangoflux_ckpt", type=str, default="", help="Specify path to TangoFlux checkpoint and configs (.ckpt, .json, etc..)")
    args = parser.parse_args()
    
    DEV_DATASET_ROOT = Path(args.dataset_folder)
    OUTPUT_ROOT = Path(args.output_folder)
    
    audio_folder = OUTPUT_ROOT.joinpath("waveforms")
    audio_folder.mkdir(parents=True, exist_ok=False)
    print(f"Output directory created at: {audio_folder}")
    
    # Load models
    print("############## Loading AudioLDM ##############")
    audioldm_model = AudioldmModel() 
    audioldm_model.load(args.audioldm_ckpt) # no filepath loads the default pre-trained model
    print("############## Loading TangoFlux ##############")
    tango_flux_model = TangoFluxModel()
    tango_flux_model.load(args.tangoflux_ckpt) # no filepath loads the default pre-trained model
    
    # Get DCASE development set prompts
    captions = parse_csv(DEV_DATASET_ROOT.joinpath("captions.csv"))
    print ("Loaded all development captions...", captions[-1])

    
    for filename, sentence in captions:
        print (f"Generating: {filename} | {sentence}")

        audioldm_audio = audioldm_model.infer(prompt=sentence, duration=4.0)
        tango_audio = tango_flux_model.infer(prompt=sentence, duration=4.0)

        # Save the audio file to the output folder
        torchaudio.save(audio_folder.joinpath("audioldm_" + filename), audioldm_audio, sample_rate=SAMPLE_RATE)
        torchaudio.save(audio_folder.joinpath("tangoflux_" + filename), audioldm_audio, sample_rate=SAMPLE_RATE)
    
    print("############################ Finished inference on both models!")
        