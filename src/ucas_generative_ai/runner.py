"""Run the recorded protocols into a separate user-selected results directory."""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_model, load_state_dict, make_model
from .config import EXPERIMENTS, experiment_config, seed_all
from .data import load_mnist
from .paths import ensure_run_output
from .training import TrainingSession


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def initial_weights(output, explicit=None):
    if explicit is not None:
        path = Path(explicit)
    else:
        path = Path(output)/"g/pixelcnn_best.pt"
        if not path.exists():
            path = Path("checkpoints/g.pt")
    if not path.exists():
        raise FileNotFoundError("Fine-tuning requires G's best weights. Train --experiment g first, "
                                "or use --initial-checkpoint checkpoints/g.pt.")
    return path


def run_experiments(experiment, output="runs", data_root="data", device="cpu",
                    *, download=True, resume=True, initial_checkpoint=None):
    output = ensure_run_output(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    data = load_mnist(data_root, download=download)
    data.save_protocol(output)
    torch.backends.cudnn.benchmark = True
    write_json(output/"environment.json", dict(python=platform.python_version(),
        torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None))
    names = EXPERIMENTS if experiment == "all" else (experiment,)
    entries = {}
    for name in names:
        config = experiment_config(name)
        directory = output/name
        state_path = directory/"training_state.pt"
        saved_config = directory/"config.json"
        initialization = None
        if name in ("f0", "f02"):
            path = initial_weights(output, initial_checkpoint)
            initialization = load_state_dict(path)
            if "vertical_input.weight" not in initialization:
                raise ValueError("Fine-tuning must start from a gated G checkpoint.")
            recorded_metrics = path.parent/"metrics.json"
            if recorded_metrics.exists():
                config["initial_train_nll"] = json.loads(recorded_metrics.read_text())["train"]["nll_nats_per_image"]
            else:
                initial_model, initial_config = load_model(path, device=device)
                if initial_config["architecture"] != "dilated_gated":
                    raise ValueError("Fine-tuning must start from a gated G checkpoint.")
                validation_loader = data.loader("validation",128,device)
                initial_session = TrainingSession(initial_config,validation_loader,device,directory)
                initial_train, _ = initial_session.evaluate(initial_model,data.loader("train",128,device))
                config["initial_train_nll"] = initial_train["nll_nats_per_image"]
                del initial_model, initial_session
            config["initial_weights_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            g_summary = path.parent/"training_summary.json"
            if g_summary.exists():
                config["initial_epoch"] = json.loads(g_summary.read_text())["best_epoch"]
            # For the released historical g.pt, initial_epoch is the recorded 9.
        if state_path.exists():
            if not resume:
                raise FileExistsError("Existing training state: " + str(state_path) +
                                      ". Choose a new --output to start independently.")
            if not saved_config.exists() or json.loads(saved_config.read_text()) != config:
                raise ValueError("Resume configuration changed: " + str(directory))
        elif directory.exists() and any(directory.glob("pixelcnn_*.pt")):
            raise FileExistsError("Weights without a full resume state already exist in " + str(directory))
        directory.mkdir(parents=True,exist_ok=True)
        write_json(saved_config,config)
        data.save_protocol(directory)
        # Match notebook initialization and per-candidate shuffle streams.
        seed_all(config["seed"])
        training_loader = data.loader("train",config["batch_size"],device,shuffle=True,seed=config["seed"]+1)
        validation_loader = data.loader("validation",config["batch_size"],device)
        model = make_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])
        session = TrainingSession(config,validation_loader,device,directory)
        if initialization is None:
            session.train(training_loader,model,optimizer,config["epochs"],
                          output_dir=directory,resume=state_path.exists())
        else:
            session.train_finetuning(training_loader,model,optimizer,config["epochs"],directory,
                                    initialization,resume=state_path.exists())
        metrics = dict(train=session.evaluate(model,data.loader("train",config["batch_size"],device))[0],
                       validation=session.evaluate(model,validation_loader)[0],
                       **session.training_summary)
        write_json(directory/"metrics.json",metrics)
        entries[name] = dict(experiment=name,config=config,metrics=metrics,
                             checkpoint=str(directory/"pixelcnn_best.pt"))
        del model,optimizer,session
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selection = dict(selection_basis="validation_nll",experiments=list(names))
    if experiment == "all":
        standard = min(("s1","s2","s3"),key=lambda k:entries[k]["metrics"]["best_val_nll"])
        architecture = "g" if entries["g"]["metrics"]["best_val_nll"] < entries[standard]["metrics"]["best_val_nll"] else standard
        fine = min(("f0","f02"),key=lambda k:entries[k]["metrics"]["best_val_nll"])
        continuation = fine if entries[fine]["metrics"]["best_val_nll"] < entries["g"]["metrics"]["best_val_nll"] else "g"
        selection.update(standard=standard,architecture=architecture,continuation=continuation)
    else:
        selection["checkpoint"] = str(output/experiment/"pixelcnn_best.pt")
    # Freeze all choices before evaluating held-out test images.
    write_json(output/"selection.json",selection)
    test_names = set(names)-{"s1","s3"}
    if experiment == "all":
        test_names.add(selection["standard"])
    for name in names:
        if name not in test_names:
            continue
        config = entries[name]["config"]
        model = make_model(config).to(device)
        model.load_state_dict(load_state_dict(output/name/"pixelcnn_best.pt"))
        session = TrainingSession(config,data.loader("validation",config["batch_size"],device),device,output/name)
        metrics,values = session.evaluate(model,data.loader("test",config["batch_size"],device))
        entries[name]["metrics"]["test"] = metrics
        np.save(output/name/"test_nll_per_image.npy",values.numpy())
        write_json(output/name/"metrics.json",entries[name]["metrics"])
        del model,session
    write_json(output/"experiment_summary.json",dict(selection=selection,results=entries))
    return dict(selection=selection,results=entries)


def evaluate_checkpoint(checkpoint,data_root="data",device="cpu",*,experiment=None,
                        split="test",batch_size=128,download=True,baselines=False):
    model,config = load_model(checkpoint,experiment=experiment,device=device)
    data = load_mnist(data_root,download=download,threshold=config["threshold"],split_seed=config["split_seed"])
    session = TrainingSession(config,data.loader("validation",batch_size,device),device,Path("."))
    metrics,values = session.evaluate(model,data.loader(split,batch_size,device))
    result = dict(split=split,metrics=metrics)
    if baselines:
        train = data.train.tensors[0]
        images = getattr(data,split).tensors[0]
        probability = (train.sum(0,keepdim=True)+1)/(len(train)+2)
        independent = -(images*probability.log()+(1-images)*(1-probability).log()).flatten(1).sum(1).double()
        result["baselines"] = dict(fair_bernoulli_nll=784*np.log(2),
            independent_bernoulli_nll=independent.mean().item(),
            independent_standard_error=independent.std(unbiased=True).item()/np.sqrt(len(independent)))
    return result,values
