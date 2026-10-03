"""Raster-order Bernoulli sampling, including the first row and column."""
import math
import time
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def sample_images(model, n=64, seed=20260921, capture=False, *, device=None, height=28, width=28):
    if n < 1:
        raise ValueError("n must be positive")
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    images = torch.zeros(n,1,height,width,device=device)
    frames = [images[:8].cpu().clone()] if capture else []
    start = time.perf_counter()
    capture_rows = {height//4, height//2, 3*height//4, height}
    for row in range(height):
        for col in range(width):
            probability = model(images)[:,0,row,col]
            images[:,0,row,col] = torch.bernoulli(probability, generator=generator)
        if capture and row + 1 in capture_rows:
            frames.append(images[:8].cpu().clone())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return images.cpu(), frames, time.perf_counter()-start


def save_grid(images, path, ncols=8):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = images.detach().cpu().numpy()[:,0]
    nrows = math.ceil(len(array)/ncols)
    fig, axes = plt.subplots(nrows,ncols,figsize=(5.8,5.8*nrows/ncols),squeeze=False)
    for index, ax in enumerate(axes.flat):
        ax.set_xticks([]);ax.set_yticks([])
        if index < len(array):
            ax.imshow(array[index],cmap="gray_r",vmin=0,vmax=1,interpolation="nearest")
        else:
            ax.axis("off")
    fig.subplots_adjust(left=.005,right=.995,bottom=.005,top=.995,wspace=.06,hspace=.06)
    fig.savefig(path,dpi=200,bbox_inches="tight",pad_inches=.015)
    plt.close(fig)
