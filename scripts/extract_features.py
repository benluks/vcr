from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml

from knn_vc.hubconf import knn_vc as load_knn_vc


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, target_sr, load=True):
        self.data = Path(datafile).read_text().strip().splitlines()
        self.load = load
        self.target_sr = target_sr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx].strip()
        wav = None
        if self.load:
            wav, sr = torchaudio.load(path)
            wav = torchaudio.transforms.Resample(sr, self.target_sr)(wav)
        return path, wav


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("cfg", required=True)
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    cfg = load_hyperpyyaml(args.cfg, overrides=args.overrides)

    model = load_knn_vc(**cfg["model"])
    dataset = cfg["dataset"]

    out_path = Path(cfg["output_dir"])
    out_path.mkdir(parents=True, exist_ok=True)

    for path, _ in tqdm(dataset):
        features = model.get_features(path)
        torch.save(features.cpu(), out_path / f"{Path(path).stem}.pt")
