dependencies = ["torch", "torchaudio", "numpy"]

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import logging
import json
from pathlib import Path
from urllib.parse import urlparse


from .wavlm.WavLM import WavLM, WavLMConfig
from .hifigan.models import Generator as HiFiGAN
from .hifigan.utils import AttrDict
from .matcher import KNeighborsVC


WAVLM_CKPT = "https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt"
HIFIGAN_CKPT = (
    "https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt"
)


def knn_vc(
    pretrained=True,
    progress=True,
    prematched=True,
    device="cuda",
    hifigan_ckpt=HIFIGAN_CKPT,
    wavlm_ckpt=WAVLM_CKPT,
    layer=6
) -> KNeighborsVC:
    """Load kNN-VC (WavLM encoder and HiFiGAN decoder). Optionally use vocoder trained on `prematched` data."""
    hifigan, hifigan_cfg = hifigan_wavlm(
        pretrained, progress, prematched, device, ckpt_path=hifigan_ckpt
    )
    wavlm = wavlm_large(pretrained, progress, device, ckpt_path=wavlm_ckpt)
    knnvc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, layer, device)
    return knnvc


def hifigan_wavlm(
    pretrained=True,
    progress=True,
    prematched=True,
    device="cuda",
    ckpt_path=HIFIGAN_CKPT,
) -> HiFiGAN:
    """Load pretrained hifigan trained to vocode wavlm features. Optionally use weights trained on `prematched` data."""
    cp = Path(__file__).parent.absolute()

    with open(cp / "hifigan" / "config_v1_wavlm.json") as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    device = torch.device(device)

    generator = HiFiGAN(h).to(device)

    if pretrained:
        state_dict_g = load_ckpt(ckpt_path, device, progress=progress)
        generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()
    print(
        f"[HiFiGAN] Generator loaded with {sum([p.numel() for p in generator.parameters()]):,d} parameters."
    )
    return generator, h


def wavlm_large(
    pretrained=True, progress=True, device="cuda", ckpt_path=WAVLM_CKPT
) -> WavLM:
    """Load the WavLM large checkpoint from the original paper. See https://github.com/microsoft/unilm/tree/master/wavlm for details."""

    if is_url(ckpt_path):
        checkpoint = torch.hub.load_state_dict_from_url(
            ckpt_path, map_location=device, progress=progress
        )
    else:
        checkpoint = torch.load(ckpt_path, map_location=device)

    cfg = WavLMConfig(checkpoint["cfg"])
    device = torch.device(device)
    model = WavLM(cfg)
    if pretrained:
        model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    print(
        f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters."
    )
    return model


def is_url(s: str) -> bool:
    parsed = urlparse(s)
    return parsed.scheme in ("http", "https", "ftp") and bool(parsed.netloc)


def load_ckpt(s, device, **kwargs):
    if is_url(s):
        return torch.hub.load_state_dict_from_url(s, map_location=device, **kwargs)
    else:
        return torch.load(s, map_location=device)
