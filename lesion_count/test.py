import argparse
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

'''
JUNK fo testing
'''

def compute_eigenv3 (x):

    #print(assembled_hessian.shape)
    #print('Original Value: ', x[0][1][2][2][2].item())
    #print('Reshaped Value: ', assembled_hessian[2][2][2][1][0].item())
    #print(assembled_hessian)


    eigenvalues = torch.linalg.eigvals(x).real
    #print(eigenvalues.shape)
    #print(eigenvalues)
    return eigenvalues



def compute_hessian3(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return torch.from_numpy(hessian)


def compute_hessian2(x, diagonal=0):
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

    
    hessian_kernel = torch.stack([dxx,dxy,dxz,-dxy,dyy,dyz,-dxz,-dyz,dzz]).unsqueeze(1).cuda()
    #print(hessian_kernel.shape)
    return torch.nn.functional.conv3d(x.unsqueeze(0).unsqueeze(1), hessian_kernel, padding=1)


x = np.random.randn(100, 100, 100)
print(compute_hessian3(x).shape)



