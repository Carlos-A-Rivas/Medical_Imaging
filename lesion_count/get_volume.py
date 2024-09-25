import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import os

import skimage
from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity

from tqdm import tqdm
import time
import warnings


def main():
    parser = argparse.ArgumentParser(description="Perform lesion count processing.")
    parser.add_argument('img_path', type=str, help='Path to the file for desired volume')
    inputs = parser.parse_args()

    img_np = nib.load(inputs.img_path).get_fdata()
    volume = np.sum(img_np > 0) * .001

    print(f"\n{volume}")

main()