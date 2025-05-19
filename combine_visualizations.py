import os
import sys
from PIL import Image
import argparse

def get_png_files(directory):
    """Returns a set of PNG filenames (without path) in the given directory."""
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return None
    png_files = set()
    for f in os.listdir(directory):
        if f.lower().endswith(".png"):
            png_files.add(f)
    return png_files

def combine_images_horizontally(img1_path, img2_path, output_path):
    """
    Opens two images, pastes them side-by-side, and saves to output_path.
    """
    try:
        with Image.open(img1_path) as img1, Image.open(img2_path) as img2:
            # Ensure both images are in RGBA mode to handle transparency consistently
            # If one is RGB and other RGBA, pasting can cause issues.
            img1 = img1.convert("RGBA")
            img2 = img2.convert("RGBA")

            # Determine dimensions for the new image
            # New width is sum of both image widths
            # New height is the max of the two image heights
            dst_width = img1.width + img2.width
            dst_height = max(img1.height, img2.height)

            # Create a new image with a transparent background
            dst_image = Image.new('RGBA', (dst_width, dst_height), (0, 0, 0, 0))

            # Paste the first image
            dst_image.paste(img1, (0, 0))

            # Paste the second image next to the first
            dst_image.paste(img2, (img1.width, 0))

            # Save the combined image
            dst_image.save(output_path)
            print(f"Successfully combined '{os.path.basename(img1_path)}' and saved to '{output_path}'")

    except FileNotFoundError:
        print(f"Error: One or both image files not found: '{img1_path}', '{img2_path}'")
    except Exception as e:
        print(f"Error processing images '{os.path.basename(img1_path)}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Combine pairs of PNG images from two directories side-by-side.")
    parser.add_argument("dir1", help="Path to the first input directory.")
    parser.add_argument("dir2", help="Path to the second input directory.")
    parser.add_argument("out_dir", help="Path to the output directory for combined images.")
    
    args = parser.parse_args()

    dir1_path = args.dir1
    dir2_path = args.dir2
    output_dir_path = args.out_dir

    # 1. Get PNG files from both directories
    print(f"Scanning directory 1: {dir1_path}")
    pngs_dir1 = get_png_files(dir1_path)
    if pngs_dir1 is None:
        sys.exit(1)

    print(f"Scanning directory 2: {dir2_path}")
    pngs_dir2 = get_png_files(dir2_path)
    if pngs_dir2 is None:
        sys.exit(1)

    if not pngs_dir1:
        print(f"No PNG files found in '{dir1_path}'. Exiting.")
        sys.exit(0)
    if not pngs_dir2:
        print(f"No PNG files found in '{dir2_path}'. Exiting.")
        sys.exit(0)

    # 2. Assert that the two directories contain the same PNG files
    assert pngs_dir1 == pngs_dir2

    # 3. Create output directory if it doesn't exist
    if not os.path.exists(output_dir_path):
        try:
            os.makedirs(output_dir_path)
            print(f"Created output directory: '{output_dir_path}'")
        except OSError as e:
            print(f"Error: Could not create output directory '{output_dir_path}': {e}")
            sys.exit(1)
    elif not os.path.isdir(output_dir_path):
        print(f"Error: Output path '{output_dir_path}' exists but is not a directory.")
        sys.exit(1)
    else:
        print(f"Output directory '{output_dir_path}' already exists. Files will be overwritten if names match.")

    # 4. Process each pair of PNG files
    print(f"\nProcessing {len(pngs_dir1)} image pair(s)...")
    for filename in sorted(list(pngs_dir1)): # Iterate through one set, as they are identical
        img1_full_path = os.path.join(dir1_path, filename)
        img2_full_path = os.path.join(dir2_path, filename)
        output_full_path = os.path.join(output_dir_path, filename)

        combine_images_horizontally(img1_full_path, img2_full_path, output_full_path)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()