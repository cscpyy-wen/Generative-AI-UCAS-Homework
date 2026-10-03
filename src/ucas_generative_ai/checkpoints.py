"""Load released state dictionaries without requiring training-state archives."""
import json
from pathlib import Path

import torch

from .config import EXPERIMENTS, experiment_config
from .models import PixelCNN, GatedPixelCNN, RegularizedGatedPixelCNN


def load_state_dict(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch 1.12 predates the weights_only keyword.
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Expected a plain model state_dict, not a full training-state file.")
    return state


def infer_config(path, state, experiment=None):
    path = Path(path)
    if experiment:
        return experiment_config(experiment)
    sidecar = path.parent/"config.json"
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    if path.stem.lower() in EXPERIMENTS:
        return experiment_config(path.stem.lower())
    if "input_conv.weight" in state:
        config = experiment_config("s2")
        config["channels"] = state["input_conv.weight"].shape[0]
    elif "vertical_input.weight" in state:
        config = experiment_config("g")
        config["channels"] = state["vertical_input.weight"].shape[0]//2
    else:
        raise ValueError("Unknown checkpoint architecture; supply --experiment.")
    config["n_blocks"] = len({key.split(".")[1] for key in state if key.startswith("blocks.")})
    if config["architecture"] == "dilated_gated":
        config["dilations"] = ([1,2,4]*4)[:config["n_blocks"]]
    return config


def make_model(config):
    if config["architecture"] == "standard":
        return PixelCNN(config["channels"], config["n_blocks"])
    args = (config["channels"], config["n_blocks"], config["dilations"])
    if "dropout_p" in config:
        return RegularizedGatedPixelCNN(*args, dropout_p=config["dropout_p"])
    return GatedPixelCNN(*args)


def load_model(path, *, experiment=None, device="cpu"):
    state = load_state_dict(path)
    config = infer_config(path, state, experiment)
    model = make_model(config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config
