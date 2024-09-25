import argparse
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

def generate_gaussian_blob(shape, center, size, intensity):
    """Generate a 3D Gaussian blob with scaling"""
    grid = np.indices(shape)
    grid = [g - c for g, c in zip(grid, center)]
    blob = np.exp(-sum((g / size) ** 2 for g in grid) / 2)
    return intensity * blob

def create_synthetic_image(dimensions, lesion_sizes, lesion_intensities, lesion_distance):
    image = np.zeros(dimensions)
    
    # Calculate centers for the lesions
    center = np.array(dimensions) // 2
    center1 = center.copy()
    center2 = center.copy()
    center1[2] -= lesion_distance // 2  # Change from x-axis to z-axis
    center2[2] += lesion_distance // 2  # Change from x-axis to z-axis
    
    # Generate lesions with varying intensities
    lesion1 = generate_gaussian_blob(dimensions, center1, lesion_sizes[0], lesion_intensities[0])
    lesion2 = generate_gaussian_blob(dimensions, center2, lesion_sizes[1], lesion_intensities[1])
    
    # Combine lesions into the image
    image += lesion1
    image += lesion2
    
    # Normalize to ensure values are between 0 and 1
    image = np.clip(image, 0, 1)
    
    return image

def main():
    parser = argparse.ArgumentParser(description='Synthetic MS Lesion Generator')
    parser.add_argument('--dimensions', type=int, nargs=3, required=True, help='Dimensions of the synthetic image (x, y, z)')
    parser.add_argument('--lesion_sizes', type=float, nargs=2, required=True, help='Sizes of the two lesions (standard deviations)')
    parser.add_argument('--lesion_intensities', type=float, nargs=2, required=True, help='Intensities of the two lesions')
    parser.add_argument('--lesion_distance', type=int, required=True, help='Total distance between the two lesions along the z-axis')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder path')
    
    args = parser.parse_args()

    synthetic_image = create_synthetic_image(args.dimensions, args.lesion_sizes, args.lesion_intensities, args.lesion_distance)
    
    # Generate file name based on parameters
    file_name = f"{args.dimensions[0]}x{args.dimensions[1]}x{args.dimensions[2]}_" \
                f"sizes_{args.lesion_sizes[0]}_{args.lesion_sizes[1]}_" \
                f"intensities_{args.lesion_intensities[0]}_{args.lesion_intensities[1]}_" \
                f"distance_{args.lesion_distance}.nii"
    
    output_path = os.path.join(args.output_folder, file_name)
    
    # Create NIfTI image
    nifti_image = nib.Nifti1Image(synthetic_image, np.eye(4))
    nib.save(nifti_image, output_path)
    print(f"Synthetic image saved as {output_path}")

if __name__ == "__main__":
    main()
