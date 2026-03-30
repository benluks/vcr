from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml

from knn_vc.hubconf import knn_vc as load_knn_vc

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("cfg")
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = load_hyperpyyaml(f, overrides=args.overrides)

    model = load_knn_vc(**cfg["model"])
    dataset = cfg["data"]

    out_path = Path(cfg["output_dir"])
    out_path.mkdir(parents=True, exist_ok=True)

    for path, _ in tqdm(dataset):
        features = model.get_features(path)
        torch.save(features.cpu(), out_path / f"{Path(path).stem}.pt")
