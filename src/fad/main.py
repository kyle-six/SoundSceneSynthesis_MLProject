import glob, os, tqdm, argparse
import numpy as np
from pathlib import Path

def load_embeddings(emb_dir):
    files = sorted(Path(emb_dir).rglob("*.npy"))
    return np.vstack([np.load(str(f)) for f in files])

def stable_frechet_distance(emb1, emb2, eps=1e-6):
    # means
    mu1, mu2 = emb1.mean(0), emb2.mean(0)
    diff = mu1 - mu2

    # covariances
    cov1 = np.cov(emb1, rowvar=False)
    cov2 = np.cov(emb2, rowvar=False)

    # eigen-decompose cov1·cov2
    prod = cov1.dot(cov2)
    eigvals, eigvecs = np.linalg.eig(prod)

    # clamp negatives to zero, take sqrt
    sqrt_eigvals = np.sqrt(np.clip(np.real(eigvals), 0, None))

    # reconstruct the sqrt matrix
    covmean = (eigvecs * sqrt_eigvals) @ np.linalg.inv(eigvecs)
    covmean = np.real(covmean)
    # symmetrize
    covmean = (covmean + covmean.T) / 2

    # compute FAD
    return diff.dot(diff) + np.trace(cov1 + cov2 - 2*covmean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="", help=" Directory containing the .wav, .csv, and .npy emdeddings from the DCASE development dataset")
    parser.add_argument("--output_folder", type=str, default="", help=" Outer directory where all audio, embeddings and more have been generated")
    parser.add_argument("--selected_audio_folder", type=str, default="", help=" Directory where the selectively replaced audio and embeddings were saved")
    args = parser.parse_args()
    
    DEV_DATASET_FOLDER = Path(args.dataset_folder)
    OUTPUT_FODLER = Path(args.output_folder)
    SELECTED_AUDIO_FOLDER = Path(args.selected_audio_folder)
    
    emb1 = load_embeddings(DEV_DATASET_FOLDER.joinpath("panns-wavegram-logmel"))
    emb2 = load_embeddings(SELECTED_AUDIO_FOLDER.joinpath("panns-wavegram-logmel"))
    fad = stable_frechet_distance(emb1, emb2)
    
    np.save(OUTPUT_FODLER.joinpath("FINAL_FAD_SCORE.npy"), fad)
    print(f"Stable FAD: {fad:.4f}")

