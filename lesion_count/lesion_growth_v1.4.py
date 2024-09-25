# ---------------------------------------------------------------------------------------
'''
TRY RUNNING GROWTH AT A LOW THRESHOLD, THEN CROPPING OFF EXTRA VOLUME USING
PREFERED THRESHOLD MASK FROM PROB MAP
'''
# ---------------------------------------------------------------------------------------

# Lesion growth program...
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

'''
This program was designed to run entire 3D random walk volumes. HOPEFULLY fixed the warning issue with this version.
CONSIDER REMOVING LABEL 1 FROM THE FINAL IMAGE
'''


version = '1.4'
root_path = '/home/carlos/Data/proposed_outputs/growth_output'


# ---------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# ---------------------------------------------------------------------------------------

def create_png_tensor(tensor, name):
    fig,axs = plt.subplots(nrows = 1, ncols = 3, figsize = (9,3))
    axs[0].imshow(tensor[tensor.shape[0]//2].cpu().detach().numpy(), cmap = 'Greys_r', vmin = tensor.min(), vmax = tensor.max())
    axs[1].imshow(tensor[:, tensor.shape[1]//2].cpu().detach().numpy(), cmap = 'Greys_r', vmin = tensor.min(), vmax = tensor.max())
    axs[2].imshow(tensor[:, :, tensor.shape[2]//2].cpu().detach().numpy(), cmap = 'Greys_r', vmin = tensor.min(), vmax = tensor.max())

    for ax in axs.flat:
        ax.set_yticks([])
        ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(name + '.png')


def create_png_numpy(numpy, name):
    fig,axs = plt.subplots(nrows = 1, ncols = 3, figsize = (9,3))
    axs[0].imshow(numpy[numpy.shape[0]//2], cmap = 'Greys_r', vmin = numpy.min(), vmax = numpy.max())
    axs[1].imshow(numpy[:, numpy.shape[1]//2], cmap = 'Greys_r', vmin = numpy.min(), vmax = numpy.max())
    axs[2].imshow(numpy[:, :, numpy.shape[2]//2], cmap = 'Greys_r', vmin = numpy.min(), vmax = numpy.max())

    for ax in axs.flat:
        ax.set_yticks([])
        ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(name + '.png')


def create_png_numpy_color(numpy_data, name, colprof):
    if numpy_data.ndim == 2:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
        axs.imshow(numpy_data, cmap=colprof, vmin=numpy_data.min(), vmax=numpy_data.max())
        axs.set_yticks([])
        axs.set_xticks([])
    elif numpy_data.ndim == 3:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
        axs[0].imshow(numpy_data[numpy_data.shape[0]//2], cmap=colprof, vmin=numpy_data.min(), vmax=numpy_data.max())
        axs[1].imshow(numpy_data[:, numpy_data.shape[1]//2], cmap=colprof, vmin=numpy_data.min(), vmax=numpy_data.max())
        axs[2].imshow(numpy_data[:, :, numpy_data.shape[2]//2], cmap=colprof, vmin=numpy_data.min(), vmax=numpy_data.max())
        
        for ax in axs.flat:
            ax.set_yticks([])
            ax.set_xticks([])
    else:
        raise ValueError("Input numpy array must be either 2D or 3D.")

    plt.tight_layout()
    plt.savefig(name + '.png')

# ---------------------------------------------------------------------------------------


def main():

    
    # VERIFIED FUNCTIONAL
    # ---------------------------------------------------------------------------------------
    # Input argument handling
    # ---------------------------------------------------------------------------------------
    # NEW ARGUMENTS TO CONSIDER!!:
    # connnected components connectivity value
    parser = argparse.ArgumentParser(description="Perform lesion count processing.")
    parser.add_argument('flair_path', type=str, help='Path to the OG flair image')
    parser.add_argument('label_path', type=str, help='labeled image path')
    parser.add_argument('--patient', type=str, default='', help='Patient #')
    parser.add_argument('--beta', type=int, default=200, help='Higher values make the walker less likely to cross sharp gradients')
    parser.add_argument('--mode', type=str, default='cg_j', help='"bf" brute forces the computation, "cg" uses conjugate gradient')
    parser.add_argument('--tolerance', type=float, default=1e-3, help='not really sure what this parameter does, function default is 1e-3')
    parser.add_argument('--slice_width', type=int, default = 1, help='Sets the thickness for each slice computed for random walk. Must be an odd integer.')
    parser.add_argument('--thr_val', type=float, default=0.2, help='Probability mask threshold value')

    inputs = parser.parse_args()


    # Start the timer
    start_time = time.time()
    # loads the probability map
    flair_np = nib.load(inputs.flair_path).get_fdata()
    # loads lesion center file
    labels_np = nib.load(inputs.label_path).get_fdata()
    thr_val = inputs.thr_val

    
    # Threshold the prob. map
    thr_mask = (flair_np > thr_val).astype(np.uint8) # consider switching to np.where()
    flair_np[flair_np < .1] = 0 # CONSIDER INCREASING .1 IF RUNTIME IS SLOW, DECREASING IF LOWER THRESHOLD IS DESIRED
    # Create a mask for zero elements in the first image
    zero_mask = flair_np == 0
    # Create a mask for non-zero elements in the first image
    non_zero_mask = flair_np != 0
    create_png_numpy(flair_np, '/home/carlos/lesion_count/verification/flair')
    # Create a mask for zero elements in the second image
    labels_zero_mask = labels_np == 0
    # Set elements in the processed image to 0 where the first image has non-zero probabilities
    labels_np[non_zero_mask & labels_zero_mask] = 0
    # Set elements in the processed image to -1 where the first image has zero probabilities
    labels_np[zero_mask] = -1

    
    # Saving labels_np so I can look at it
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(labels_np, affine)
    nib.save(nifti_img, root_path + '/labels_np.nii')
    

    segmentation = random_walker(flair_np, labels_np, beta=inputs.beta, mode=inputs.mode, tol=inputs.tolerance)
    
    # ---------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------
    # Perform Random Walk 2D Slice-by-Slice
    # ---------------------------------------------------------------------------------------
    #segmentation = random_walker(flair_np, labels_np, beta=inputs.beta, mode=inputs.mode, tol=inputs.tolerance)

    assembled = segmentation * thr_mask
    #assembled[assembled < 0] = 0
    lesion_volume = np.sum(assembled > 0) * .001

    #segmentation = random_walker(flair_np, labels_np, beta=inputs.beta, mode=inputs.mode, tol=inputs.tolerance)
    print('Segmentation Shape: ', assembled.shape)
    create_png_numpy_color(assembled, '/home/carlos/lesion_count/verification/growth', 'jet')
    # ---------------------------------------------------------------------------------------
    
    '''
    # ---------------------------------------------------------------------------------------
    # Perform Random Walk 3D
    # ---------------------------------------------------------------------------------------
    print('Computing Random Walk...')
    print('inputs.tolerance type: ', type(inputs.tolerance))
    create_png_numpy_color(labels_np[:][:][101], '/home/carlos/lesion_count/verification/labels', 'jet')
    # ---------------------------------------------------------------------------------------
    assembled = random_walker(flair_np, labels_np, beta=inputs.beta, mode=inputs.mode, tol=inputs.tolerance)
    # ---------------------------------------------------------------------------------------
    print('Segmentation Shape: ', assembled.shape)
    create_png_numpy_color(assembled, '/home/carlos/lesion_count/verification/growth', 'jet')
    # ---------------------------------------------------------------------------------------
    '''
    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)




    # ---------------------------------------------------------------------------------------
    # Convert array to NIfTI file
    # ---------------------------------------------------------------------------------------
    # Create an affine transformation matrix (identity matrix is used if no specific transformation is needed)
    affine = np.eye(4)
    # Create the NIfTI image
    nifti_img = nib.Nifti1Image(assembled, affine)
    # Save the NIfTI image to a file
    #os.makedirs((root_path + '/growthv' + version + '_files'), exist_ok=True)
    #nib.save(nifti_img, root_path + '/growthv' + version + '_files/' + ('growth' + version + '_' + 'beta' + str(inputs.beta) + '_thr' + str(inputs.thr_val) + '.nii'))
    nib.save(nifti_img, root_path + ('/growth' + version + '_P'+ inputs.patient + '_' + str(inputs.mode) + '_thr' + str(thr_val) + '.nii'))
    # ---------------------------------------------------------------------------------------


    # I put these here just incase the print statements fail, we will still get the NIfTI output
    print(f"Total 3D Random Walk Runtime: {hours}h {minutes}m {seconds}s")


    print(f"Final Threshold Value: {thr_val}")
    print(f"\n{lesion_volume}")


main()