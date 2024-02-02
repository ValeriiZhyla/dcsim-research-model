import os
from PIL import Image
from PIL.Image import Palette

# Directory containing your PNG files
#directory = "path/to/your/directory"
#directory = r"D:\KIT\Master Thesis\dcsim-research-model\training\3rd-phase\plots"
#directory = r"D:\KIT\Master Thesis\dcsim-research-model\training\2nd-phase\plots"

directory = r"D:\KIT\Master Thesis\dcsim-research-model\trained-models\1st-phase"


# Iterate over all subdirectories and files in the directory
for root, dirs, files in os.walk(directory):
    # Check if 'plots' directory is found
    if 'plots' in root:
        # Path for the 'plots_reduced' directory, created at the same level as 'plots'
        reduced_dir = os.path.join(root, 'plots_reduced')



        # Process all PNG files in the 'plots' directory and subdirectories
        for filename in files:
            if filename.endswith(".png"):
                file_path = os.path.join(root, filename)

                # Path for the reduced file in 'plots_reduced'
                reduced_file_path = os.path.join(reduced_dir, filename)

                # Check if reduced directory exists, create if not
                if not os.path.exists(reduced_dir):
                    os.makedirs(reduced_dir)

                # Open the image
                with Image.open(file_path) as img:
                    # Convert the image to 8-bit (256 colors)
                    img = img.convert("P", palette=Palette.ADAPTIVE)

                    # Save the image to the reduced directory with compression
                    img.save(reduced_file_path, format='PNG', compress_level=6)

print("Conversion and compression complete for all PNG files in the plots directories.")