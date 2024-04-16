import os
from PIL import Image


def minimize_image_file(file_path):
    # Open the image
    with Image.open(file_path) as img:

        # Convert the image to optimize size, or keep as is if specific conversion isn't needed
        img = img.convert("P", palette=Image.ADAPTIVE)

        # Save the image with optimization, directly overwriting the original file
        img.save(file_path, format='PNG', optimize=True, compress_level=6)


def minimize_pngs_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".png"):
                file_path = os.path.join(root, filename)

                # Minimize the image file, directly overwriting the original
                minimize_image_file(file_path)

    print("Minimization and overwriting complete for all PNG files.")


# Usage example, replace 'your_root_directory' with your actual directory path
minimize_pngs_in_directory('C:/Users/User/Desktop/Thesis/figures')