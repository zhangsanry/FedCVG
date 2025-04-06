import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

def show_images(image_dir):
    """Display all images in the specified directory"""
    # Get all PNG files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    if not image_files:
        print(f"No PNG images found in directory {image_dir}")
        return
    
    # Print the image files found
    print(f"Found the following image files:")
    for i, img_file in enumerate(image_files):
        print(f"{i+1}. {img_file}")
    
    # Print image paths for easy access by users
    print("\nYou can find the generated images at the following paths:")
    for img_file in image_files:
        full_path = os.path.abspath(os.path.join(image_dir, img_file))
        print(full_path)

if __name__ == "__main__":
    # Default directory is figures/comm_vs_accuracy
    image_dir = 'figures/comm_vs_accuracy'
    
    # If command line argument is provided, use it as the directory
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    
    show_images(image_dir) 