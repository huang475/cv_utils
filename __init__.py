# %%
from os import walk, path, remove

import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np
import matplotlib.pyplot as plt

def filenames_in_dir(d):
    filenames = []
    for _, _, _filenames in walk(d):
        filenames += (_filenames)
    return filenames
    

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(12, 4))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image.squeeze())
    plt.show()
