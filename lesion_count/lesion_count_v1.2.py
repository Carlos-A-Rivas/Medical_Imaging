import argparse
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import os

'''
THIS PROGRAM RUNS THE 5X5 MATRIX
'''

version = '1.2'

# ---------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# ---------------------------------------------------------------------------------------

def compute_hessian2(x, diagonal=0):
    '''
    dxx = torch.tensor([[[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]],

                        [[0, 0, 0],
                         [1, -2, 1],
                         [0, 0, 0]],

                        [[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]]])
    
    dyy = torch.tensor([[[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]],

                        [[0, 1, 0],
                         [0, -2, 0],
                         [0, 1, 0]],

                        [[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]]])
    
    dzz = torch.tensor([[[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]],

                        [[0, 0, 0],
                         [0, -2, 0],
                         [0, 0, 0]],

                        [[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]]])
    '''
    

    dxx = torch.tensor([[[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],

                         [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],

                        [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [.25, 0, -.5, 0, .25],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],

                         [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],

                        [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]])
    
    dyy = torch.tensor([[[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],

                         [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],

                        [[0, 0, .25, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, -.5, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, .25, 0, 0]],

                         [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],

                        [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]])
    
    dzz = torch.tensor([[[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, .25, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],

                         [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],

                        [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, -.5, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],

                         [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],

                        [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, .25, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]])

    if (diagonal == 'True'):
        print('Running Diagonal Matrix')
        dxy = torch.tensor([[[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],

                            [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],

                            [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]], dtype=torch.float32)
        
        dxz = torch.tensor([[[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],

                            [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],

                            [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]], dtype=torch.float32)
        
        dyz = torch.tensor([[[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],

                            [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],

                            [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]], dtype=torch.float32)
    else:
        print('Running Hessian Matrix')
        dxy = torch.tensor([[[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],

                            [[.25, 0, -.25],
                            [0, 0, 0],
                            [-.25, 0, .25]],

                            [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]], dtype=torch.float32)
        
        dxz = torch.tensor([[[0, 0, 0],
                            [.25, 0, -.25],
                            [0, 0, 0]],

                            [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],

                            [[0, 0, 0],
                            [-.25, 0, .25],
                            [0, 0, 0]]], dtype=torch.float32)
        
        dyz = torch.tensor([[[0, .25, 0],
                            [0, 0, 0],
                            [0, -.25, 0]],

                            [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],

                            [[0, -.25, 0],
                            [0, 0, 0],
                            [0, .25, 0]]], dtype=torch.float32)
        
        padding_pattern = (1, 1, 1, 1, 1, 1)
        dxy = torch.nn.functional.pad(dxy, padding_pattern, "constant", 0)
        dxz = torch.nn.functional.pad(dxz, padding_pattern, "constant", 0)
        dyz = torch.nn.functional.pad(dyz, padding_pattern, "constant", 0)

    
    hessian_kernel = torch.stack([dxx,dxy,dxz,dxy,dyy,dyz,dxz,dyz,dzz]).unsqueeze(1).cuda()
    #print(hessian_kernel.shape)
    return torch.nn.functional.conv3d(x.unsqueeze(0).unsqueeze(1), hessian_kernel, padding=2)


def compute_eigenv2 (x):
    '''
    print(hessian2[1][2][3][4][5])
    1 controls the which batch you are working with (I think)
    2 controls which partial you are working with (Dxx=0, Dxy=1, Dxz=2 etc.)
    3 Z coordinate in 3D space
    4 Y coordinate in 3D space
    5 X coordinate in 3D space
    '''
    print("\nRunning compute_eigen2")

    # RESHAPE INTO NEW DIMENSIONS OF PARTIAL SIZES
    '''
    (after operation)
    print(assembled_hessian[1][2][3][4][5])
    1 Z coordinate in 3D space
    2 Y coordinate in 3D space
    3 X coordinate in 3D space
    4 x axis of hessian matrix
    5 y axis of hessian matrix
    '''
    assembled_hessian = x.squeeze(0).view(3,3,x[0][0].shape[0],x[0][0].shape[1],x[0][0].shape[2]).permute(2,3,4,0,1)

    #print(assembled_hessian.shape)
    #print('Original Value: ', x[0][1][2][2][2].item())
    #print('Reshaped Value: ', assembled_hessian[2][2][2][1][0].item())
    #print(assembled_hessian)


    eigenvalues = torch.linalg.eigvals(assembled_hessian).real
    #print(eigenvalues.shape)
    #print(eigenvalues)
    return eigenvalues


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
    # ---------------------------------------------------------------------------------------
    # HEADER CREATION
    # ---------------------------------------------------------------------------------------
    print('\n\n')
    for i in range(30):
        print('||', end='')
    print('\nBeginning Lesion Count Program...')
    for i in range(30):
        print('--', end='')
    print('\n')
    # ---------------------------------------------------------------------------------------


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
    parser.add_argument('--diagonal', type=str, default='False', help='Chooses between normal hessian matrix or diagonal matrix. True input uses diagonal.')
    parser.add_argument('--connectivity', type=int, default=26, help='Choose between 6, 18, and 26 point connectivity')
    parser.add_argument('--eigen_thr', type=float, default=0.0, help='Determines threshold where eigen locations are marked. Default is values below 0.')
    #parser.add_argument('--output_name', type=str, default='lesion_count_mask', help='Output file name')
    parser.add_argument('--gamma_correction', type=str, default='False', help='Turns gamma correction on or off')
    parser.add_argument('--gamma_val', type=float, default=0.0, help='Sets gamma value')
    parser.add_argument('--slice_removal', type=str, default='False', help='Turns slice removal on or off')
    parser.add_argument('--removal_val', type=int, default=1, help='Sets size threshold at which a lesion would be removed')
    # Good Color Profiles: magma, jet
    parser.add_argument('--color_profile', type=str, default='jet', help='Changes PNG color profile')
    parser.add_argument('--output_folder', type=str, default='/home/carlos/Data/proposed_outputs/lesion_centers', help='output folder path for files created; do not include a "/" at the end of the path')
    inputs = parser.parse_args()

    print('inputs.diagonal: ', inputs.diagonal)
    print('inputs.gamma_correction: ', inputs.gamma_correction)

    # Convert input probability map to a pytorch tensor (an array with special methods that can be called)
    ples_np = nib.load(inputs.ples_path).get_fdata()
    create_png_numpy(ples_np, 'original_img')
    #print('ples_np variable type --> ', type(ples_np))
    ples_tensor = torch.tensor(ples_np, dtype = torch.float32, device='cuda')
    #print('ples_tensor variable type --> ', type(ples_tensor), '\n')
    # ---------------------------------------------------------------------------------------


    # VERIFIED FUNCTIONAL
    # ---------------------------------------------------------------------------------------
    # Gamma Correction Filter
    # ---------------------------------------------------------------------------------------
    if (inputs.gamma_correction == 'True'):
        # Gamma correction step
        print('Gamma Value: ', inputs.gamma_val)
        ples_tensor = ples_tensor ** inputs.gamma_val
    # ---------------------------------------------------------------------------------------


    #  VERIFIED FUNCTIONAL
    # ---------------------------------------------------------------------------------------
    # Goal is to compute the eigenvalues at each voxel, and create a new map that only includes
    # the voxels where all eigenvalues were zero
    # ---------------------------------------------------------------------------------------
    if (inputs.connectivity == 6):
        connectivity = np.array([[[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]],
                                
                                [[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]],
                                
                                [[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]]])
    elif (inputs.connectivity == 18):
        connectivity = np.array([[[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]],
                                
                                [[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]],
                                
                                [[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]]])
    elif (inputs.connectivity == 26):
        connectivity = np.array([[[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]],
                                
                                [[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]],
                                
                                [[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]])
    else:
        print('\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nINVALID CONNECTIVITY VALUE ENTERED, USING DEFAULT VALUE OF ', inputs.connectivity, '\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n')
        connectivity = np.array([[[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]],
                                
                                [[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]],
                                
                                [[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]])
        
    eigen_matrix = compute_eigenv2(compute_hessian2(ples_tensor, inputs.diagonal))
    print('Eigen Matrix Shape: ', eigen_matrix.shape, '\n')
    # Step 1: Check if all eigenvalues in the last dimension are negative
    ## CHANGES EIGENVALUE KEEP THRESHOLD
    negative_eigen_locations = (eigen_matrix < inputs.eigen_thr).all(dim=-1).int()  # Returns a [X, Y, Z] tensor of booleans, .int() converts to 1s or 0s
    create_png_tensor(negative_eigen_locations, 'negative_eigen_locations')
    # This section takes all of the negative eigen locations and removes locations below a certain probability
    threshold_mask = ples_tensor > inputs.thr_val
    thr_negative_eigen_locations = negative_eigen_locations * threshold_mask
    # ----------------------------------------------------------------------------------------------------------
    create_png_tensor(thr_negative_eigen_locations, 'thr_negative_eigen_locations')

    '''
    ###### REMOVE AFTER DONE TESTING ###########
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(thr_negative_eigen_locations.cpu().detach().numpy().astype(float), affine)
    nib.save(nifti_img, f"/home/carlos/lesion_count/verification/thr_negative_eigen_locations.nii")
    ###### REMOVE AFTER DONE TESTING ###########
    '''

    labeled_count, num_features = scipy.ndimage.label(thr_negative_eigen_locations.cpu().detach().numpy(), structure=connectivity)
    #print('Number of Lesions: ', num_features, '\nSegmented Count Array Dimensions: ', labeled_count_np.shape)
    print('Number of lesions from ' + str(inputs.connectivity) + '-point connectivity: ', num_features)
    create_png_numpy_color(labeled_count, 'labeled_count_' + str(inputs.connectivity), inputs.color_profile)
    



    # ---------------------------------------------------------------------------------------
    # Small Feature Removal
    # ---------------------------------------------------------------------------------------
    print('Inputs.slice_removal:', inputs.slice_removal)
    if (inputs.slice_removal == 'True'):
        print('Slice Removal Active')
        # ---------------------------------------------------------------------------------------
        # Copied from V2
        # ---------------------------------------------------------------------------------------
        # Size threshold
        # ADD PARSER FOR THIS PARAMETER
        size_threshold = inputs.removal_val

        # Find slices for each labeled component
        slices = scipy.ndimage.find_objects(labeled_count)

        # Create a mask for components above the size threshold
        mask = np.zeros(labeled_count.shape, dtype=bool)

        for i, component_slice in enumerate(slices):
            if component_slice is not None:  # Check if slice is valid
                component = labeled_count[component_slice]
                component_mask = (component == (i + 1))
                if np.sum(component_mask) >= size_threshold:
                    mask[component_slice][component_mask] = True

        # Apply the mask to the labeled matrix
        labeled_count = np.where(mask, labeled_count, 0)
        # ---------------------------------------------------------------------------------------
    
    
    # ---------------------------------------------------------------------------------------
    # Convert array to NIfTI file
    # ---------------------------------------------------------------------------------------
    print('Input Image Dimentions: ', ples_np.shape)
    print('Final Image Dimentions: ', labeled_count.shape)
    # Create an affine transformation matrix (identity matrix is used if no specific transformation is needed)
    affine = np.eye(4)
    # Create the NIfTI image
    nifti_img = nib.Nifti1Image(labeled_count, affine)
    # Save the NIfTI image to a file
    #os.makedirs((root_path + '/v' + version + '_files'), exist_ok=True)
    #nib.save(nifti_img, root_path + '/v' + version + '_files/' + ('LCV' + version + '_' + 'thr(' + str(inputs.thr_val) + ')' + 'connec(' + str(inputs.connectivity) + ')' + 'gamma(' + inputs.gamma_correction + ')' + 'gam_val(' + str(inputs.gamma_val) + ')' + '.nii'))
    #nib.save(nifti_img, root_path + '/v' + version + '_files/' + ('count' + version + '_' + 'thr' + str(inputs.thr_val) + 'connec' + str(inputs.connectivity) + 'gamma' + inputs.gamma_correction + 'gam_val' + str(inputs.gamma_val) + '.nii'))
    nib.save(nifti_img, inputs.output_folder + ('/count' + version + '_P'+ inputs.patient + '_C' + str(num_features) + '_thr' + str(inputs.thr_val) + 'connec' + str(inputs.connectivity) + 'gamma' + inputs.gamma_correction + 'gam_val' + str(inputs.gamma_val) + '.nii'))
    # ---------------------------------------------------------------------------------------
    
    '''
    nib.save(nifti_img, inputs.output_folder + ('/count' + version + '_P'+ inputs.patient + '_C' + str(num_features) + '_thr' + str(inputs.thr_val) + 'connec' + str(inputs.connectivity) + 'gamma' + inputs.gamma_correction + 'gam_val' + str(inputs.gamma_val) + '.nii'))
    f"{inputs.output_folder}"
    '''

    # ---------------------------------------------------------------------------------------
    # FOOTER CREATION
    # ---------------------------------------------------------------------------------------
    print('')
    for i in range(30):
        print('--', end='')
    print('\nLesion Count Has Finished.')
    for i in range(30):
        print('||', end='')
    print('\n')
    # ---------------------------------------------------------------------------------------

    # printing num_features last allows the number of lesions to be 
    # extracted for batch_run.py
    print(num_features)


main()