import argparse
import os
import numpy as np
import nibabel as nib

def generate_irregular_blob(shape, center, size, intensity):
    """Generate a 3D irregular blob without noise"""
    grid = np.indices(shape)
    grid = [g - c for g, c in zip(grid, center)]
    distance_squared = sum((g / size) ** 2 for g in grid)
    blob = np.exp(-distance_squared / 2)
    
    # Clip to ensure values are between 0 and 1
    blob = np.clip(blob, 0, 1)
    
    return intensity * blob

def create_synthetic_image(size_range, intensity_range):
    # Calculate size and intensity step
    size_step = (size_range[1] - size_range[0]) / 6
    intensity_step = (intensity_range[1] - intensity_range[0]) / 6
    
    # Calculate the required dimensions to fit all blobs
    max_size = size_range[1]
    padding = int(max_size * 2.5)  # Ensure enough space around the blobs to prevent overlapping
    grid_size = 7
    image_size_y = grid_size * (max_size + padding)
    image_size_z = grid_size * (max_size + padding)
    
    # Set dimensions to fit the blobs in x, y, z
    dimensions = [int(max_size + padding), int(image_size_y), int(image_size_z)]
    image = np.zeros(dimensions)
    
    # Calculate centers for the blobs
    y_positions = np.linspace(padding // 2, image_size_y - padding // 2, grid_size)
    z_positions = np.linspace(padding // 2, image_size_z - padding // 2, grid_size)
    center_x = dimensions[0] // 2
    
    for i, y in enumerate(y_positions):
        for j, z in enumerate(z_positions):
            lesion_center = [center_x, int(y), int(z)]
            size = size_range[0] + i * size_step
            intensity = intensity_range[0] + j * intensity_step
            
            # Generate lesions with irregular shapes and varying intensities
            lesion = generate_irregular_blob(dimensions, lesion_center, size, intensity)
            
            # Combine lesions into the image
            image += lesion
    
    # Normalize to ensure values are between 0 and 1
    image = np.clip(image, 0, 1)
    
    return image

def main():
    parser = argparse.ArgumentParser(description='Synthetic MS Lesion Generator')
    parser.add_argument('--size_range', type=float, nargs=2, required=True, help='Range of lesion sizes (min_size max_size)')
    parser.add_argument('--intensity_range', type=float, nargs=2, required=True, help='Range of lesion intensities (min_intensity max_intensity)')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder path')
    
    args = parser.parse_args()
    
    synthetic_image = create_synthetic_image(args.size_range, args.intensity_range)
    
    # Generate file name based on parameters
    file_name = f"lesion_size_{args.size_range[0]}_{args.size_range[1]}_" \
                f"intensity_{args.intensity_range[0]}_{args.intensity_range[1]}.nii"
    
    output_path = os.path.join(args.output_folder, file_name)
    
    # Create NIfTI image
    nifti_image = nib.Nifti1Image(synthetic_image, np.eye(4))
    nib.save(nifti_image, output_path)
    print(f"Synthetic image saved as {output_path}")

if __name__ == "__main__":
    main()
