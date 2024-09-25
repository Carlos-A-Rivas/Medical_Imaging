import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage

'''
This program runs the OG 3x3 hessian
'''

version = '1.0'
root_path = '/home/carlos/Data/blob_sled/outputs'
file_name = 'data4_binary'

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


def create_png_numpy_color(numpy, name, colprof):
    fig,axs = plt.subplots(nrows = 1, ncols = 3, figsize = (9,3))
    axs[0].imshow(numpy[numpy.shape[0]//2], cmap = colprof, vmin = numpy.min(), vmax = numpy.max())
    axs[1].imshow(numpy[:, numpy.shape[1]//2], cmap = colprof, vmin = numpy.min(), vmax = numpy.max())
    axs[2].imshow(numpy[:, :, numpy.shape[2]//2], cmap = colprof, vmin = numpy.min(), vmax = numpy.max())
    
    for ax in axs.flat:
        ax.set_yticks([])
        ax.set_xticks([])
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
    parser.add_argument('ples_path', type=str, help='Path to the input probability map file (NIfTI format)')
    parser.add_argument('--patient', type=str, default='', help='Patient #')
    parser.add_argument('--thr_val', type=float, default=0.2, help='Threshold value to binarize the probability map (default: 0.4118)')
    parser.add_argument('--output_folder', type=str, default ='/home/carlos/Data/proposed_outputs/binary_maps', help='File output folder')
    '''
    parser.add_argument('--diagonal', type=str, default='False', help='Chooses between normal hessian matrix or diagonal matrix. True input uses diagonal.')
    parser.add_argument('--connectivity', type=int, default=18, help='Choose between 6, 18, and 26 point connectivity')
    parser.add_argument('--eigen_thr', type=float, default=0.0, help='Determines threshold where eigen locations are marked. Default is values below 0.')
    #parser.add_argument('--output_name', type=str, default='lesion_count_mask', help='Output file name')
    parser.add_argument('--gamma_correction', type=str, default='True', help='Turns gamma correction on or off')
    parser.add_argument('--gamma_val', type=float, default=2.0, help='Sets gamma value')
    parser.add_argument('--slice_removal', type=str, default='False', help='Turns slice removal on or off')
    parser.add_argument('--removal_val', type=int, default=1, help='Sets size threshold at which a lesion would be removed')
    # Good Color Profiles: magma, jet
    parser.add_argument('--color_profile', type=str, default='jet', help='Changes PNG color profile')
    '''
    inputs = parser.parse_args()

    connectivity = np.ones((3,3,3))
    flair_np = nib.load(inputs.ples_path).get_fdata()
    print(f"OG Image Dimensions: {flair_np.shape}")
    th_flair_np = np.where(flair_np > inputs.thr_val, 1, 0)
    print(f"Thresholded Image Dimensions: {th_flair_np.shape}")
    labeled_count, num_features = scipy.ndimage.label(th_flair_np, structure=connectivity)
    lesion_volume = np.sum(th_flair_np > 0) * .001


    
    # ---------------------------------------------------------------------------------------
    # Convert array to NIfTI file
    # ---------------------------------------------------------------------------------------
    # Create an affine transformation matrix (identity matrix is used if no specific transformation is needed)
    affine = np.eye(4)
    # Create the NIfTI image
    nifti_img = nib.Nifti1Image(labeled_count.astype(float), affine)
    # Save the NIfTI image to a file
    nib.save(nifti_img, f"{inputs.output_folder}/bmap_P{inputs.patient}_thr{inputs.thr_val}.nii.gz")

    print(f"Binary map has been saved.\n{num_features},{lesion_volume}")
    
main()