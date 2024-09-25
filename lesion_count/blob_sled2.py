import argparse
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

def generate_irregular_blob(shape, center, size, intensity):
    """Generate a 3D irregular blob without noise"""
    grid = np.indices(shape)
    grid = [g - c for g, c in zip(grid, center)]
    distance_squared = sum((g / size) ** 2 for g in grid)
    blob = np.exp(-distance_squared / 2)
    
    # Clip to ensure values are between 0 and 1
    blob = np.clip(blob, 0, 1)
    
    return intensity * blob

def create_synthetic_image(dimensions, lesion_sizes, lesion_intensities, lesion_distances):
    image = np.zeros(dimensions)
    
    # Calculate centers for the lesions
    center = np.array(dimensions) // 2
    
    for i, (size, intensity, distance) in enumerate(zip(lesion_sizes, lesion_intensities, lesion_distances)):
        lesion_center = center.copy()
        lesion_center[2] += distance  # Change to place lesions along the z-axis
        
        # Generate lesions with irregular shapes and varying intensities
        lesion = generate_irregular_blob(dimensions, lesion_center, size, intensity)
        
        # Combine lesions into the image
        image += lesion
    
    # Normalize to ensure values are between 0 and 1
    image = np.clip(image, 0, 1)
    
    return image

def main():
    parser = argparse.ArgumentParser(description='Enhanced Synthetic MS Lesion Generator')
    parser.add_argument('--dimensions', type=int, nargs=3, required=True, help='Dimensions of the synthetic image (x, y, z)')
    parser.add_argument('--num_lesions', type=int, required=True, help='Number of lesions to generate')
    parser.add_argument('--lesion_sizes', type=float, nargs='+', required=True, help='Sizes of the lesions (standard deviations)')
    parser.add_argument('--lesion_intensities', type=float, nargs='+', required=True, help='Intensities of the lesions')
    parser.add_argument('--lesion_distances', type=int, nargs='+', required=True, help='Distances of the lesions from the center along the z-axis')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder path')
    
    args = parser.parse_args()

    if not (len(args.lesion_sizes) == len(args.lesion_intensities) == len(args.lesion_distances) == args.num_lesions):
        raise ValueError("The number of lesion sizes, intensities, and distances must match the number of lesions specified.")
    
    synthetic_image = create_synthetic_image(args.dimensions, args.lesion_sizes, args.lesion_intensities, args.lesion_distances)
    
    # Generate file name based on parameters
    file_name = f"{args.dimensions[0]}x{args.dimensions[1]}x{args.dimensions[2]}_" \
                f"lesions_{args.num_lesions}_" \
                f"sizes_{'_'.join(map(str, args.lesion_sizes))}_" \
                f"intensities_{'_'.join(map(str, args.lesion_intensities))}_" \
                f"distances_{'_'.join(map(str, args.lesion_distances))}.nii"
    
    output_path = os.path.join(args.output_folder, file_name)
    
    # Create NIfTI image
    nifti_image = nib.Nifti1Image(synthetic_image, np.eye(4))
    nib.save(nifti_image, output_path)
    print(f"Synthetic image saved as {output_path}")

if __name__ == "__main__":
    main()
