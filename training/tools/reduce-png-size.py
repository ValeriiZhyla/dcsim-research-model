import os
from PIL import Image
from PIL.Image import Palette

# Directory containing your PNG files
#directory = "path/to/your/directory"
directory = r"D:\KIT\Master Thesis\dcsim-research-model\training\3rd-phase\plots"
#directory = r"D:\KIT\Master Thesis\dcsim-research-model\training\2nd-phase\plots"
#directory = r"D:\KIT\Master Thesis\dcsim-research-model\training\1st-phase\plots"


# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        file_path = os.path.join(directory, filename)

        # Path for the reduced directory
        reduced_dir = os.path.join(directory, "reduced")

        # Check if reduced directory exists, create if not
        if not os.path.exists(reduced_dir):
            os.makedirs(reduced_dir)

        # Path for the reduced file
        reduced_file_path = os.path.join(reduced_dir, filename)

        # Open the image
        with Image.open(file_path) as img:
            # Convert the image to 8-bit (256 colors)
            img = img.convert("P", palette=Palette.ADAPTIVE)

            # Save the image to the reduced directory with compression
            img.save(reduced_file_path, format='PNG', compress_level=6)

print("Conversion and compression complete for all PNG files in the directory.")