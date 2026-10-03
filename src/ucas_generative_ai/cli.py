"""Command-line entry points; expensive real training is always explicit."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_model
from .config import EXPERIMENTS,choose_device
from .paths import ensure_run_output


def parser():
    root = argparse.ArgumentParser(description="Binary MNIST PixelCNN course experiments")
    sub = root.add_subparsers(dest="command",required=True)
    train = sub.add_parser("train",help="Train or restore the recorded experiment protocols")
    train.add_argument("--experiment",choices=EXPERIMENTS+("all",),required=True)
    train.add_argument("--output",default="runs")
    train.add_argument("--initial-checkpoint",help="Common G best state_dict for f0/f02")
    train.add_argument("--no-resume",action="store_true",help="Refuse existing training states")
    evaluate = sub.add_parser("evaluate",help="Evaluate one plain state_dict on a chosen MNIST split")
    evaluate.add_argument("--checkpoint",required=True)
    evaluate.add_argument("--experiment",choices=EXPERIMENTS)
    evaluate.add_argument("--split",choices=("train","validation","test"),default="test")
    evaluate.add_argument("--batch-size",type=int,default=128)
    evaluate.add_argument("--baselines",action="store_true")
    evaluate.add_argument("--output",help="Optional JSON file (otherwise print only)")
    sample = sub.add_parser("sample",help="Generate all raster positions from an empty canvas")
    sample.add_argument("--checkpoint",required=True)
    sample.add_argument("--experiment",choices=EXPERIMENTS)
    sample.add_argument("--output",default="samples")
    sample.add_argument("--n",type=int,default=64)
    sample.add_argument("--seed",type=int,default=20260921)
    sample.add_argument("--capture",action="store_true")
    smoke = sub.add_parser("smoke",help="Small CPU-only synthetic correctness/resume checks")
    smoke.add_argument("--output",help="Optional JSON result file")
    for command in (train,evaluate):
        command.add_argument("--data-root",default="data")
        command.add_argument("--no-download",action="store_true")
    for command in (train,evaluate,sample):
        command.add_argument("--device",default="auto",help="auto, cpu, cuda, or cuda:N")
        command.add_argument("--threads",type=int,default=min(4,os.cpu_count() or 1))
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "smoke":
        from .smoke import run_smoke
        result = run_smoke()
    else:
        if args.threads < 1:
            raise ValueError("--threads must be positive")
        torch.set_num_threads(args.threads)
        device = choose_device(args.device)
        torch.backends.cudnn.benchmark = True
        if args.command == "train":
            from .runner import run_experiments
            result = run_experiments(args.experiment,args.output,args.data_root,device,
                download=not args.no_download,resume=not args.no_resume,
                initial_checkpoint=args.initial_checkpoint)
        elif args.command == "evaluate":
            if args.output:
                ensure_run_output(args.output)
            from .runner import evaluate_checkpoint
            result,_ = evaluate_checkpoint(args.checkpoint,args.data_root,device,
                experiment=args.experiment,split=args.split,batch_size=args.batch_size,
                download=not args.no_download,baselines=args.baselines)
        else:
            from .sampling import sample_images,save_grid
            directory = ensure_run_output(args.output)
            model,_ = load_model(args.checkpoint,experiment=args.experiment,device=device)
            images,frames,seconds = sample_images(model,args.n,args.seed,args.capture,device=device)
            directory.mkdir(parents=True,exist_ok=True)
            np.save(directory/"samples.npy",images.numpy())
            save_grid(images,directory/"samples.png")
            if frames:
                np.save(directory/"sampling_frames.npy",torch.stack(frames).numpy())
            result = dict(n_samples=args.n,seed=args.seed,temperature=1.,seconds=seconds,
                          generated_foreground_fraction=images.mean().item())
            (directory/"sampling_metrics.json").write_text(json.dumps(result,indent=2))
    payload = json.dumps(result,indent=2,ensure_ascii=False)
    if args.command in ("evaluate","smoke") and args.output:
        path = ensure_run_output(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(payload)
    print(payload)
