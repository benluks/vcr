from pathlib import Path

import torch
import torchaudio


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
