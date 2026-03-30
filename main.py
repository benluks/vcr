import torch

from hyperpyyaml import load_hyperpyyaml

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KNN_VC_DIR = ROOT / "knn_vc"

sys.path.insert(0, str(KNN_VC_DIR))
sys.path.insert(0, str(ROOT))

from knn_vc.hubconf import knn_vc as load_knn_vc

with open("config/knnvc.yaml") as f:
    cfg = load_hyperpyyaml(f.read())

model = load_knn_vc(**cfg["model"])
