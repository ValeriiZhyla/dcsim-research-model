# Sequence lengths and their corresponding inference times
import pandas as pd
import seaborn
from matplotlib import pyplot as plt

sequence_lengths = [5, 10, 20, 50, 100, 250, 500, 1000, 1500, 2000, 10000]
inference_times = [0.01302, 0.01043, 0.00951, 0.0095, 0.00908, 0.01103, 0.01051, 0.01522, 0.01899, 0.01702, 0.04477]

data = {
    'Sequence Length': sequence_lengths,
    'Inference Time': inference_times
}
df = pd.DataFrame(data)

# Creating the line plot
plt.figure(figsize=(10, 6))
seaborn.lineplot(data=df, x='Sequence Length', y='Inference Time', color='green', marker='o', dashes=False)

# Setting the title and labels
plt.title('Inference Time vs. Sequence Length', fontsize=16)
plt.xlabel('Sequence Length', fontsize=16)
plt.ylabel('Inference Time (s)', fontsize=16)

# Showing the plot
plt.grid(True)
plt.xscale('log')  # Using a logarithmic scale for the x-axis to better display wide-ranging values
plt.tight_layout()

plt.savefig('plots/length_vs_preparation time.png', dpi=200, bbox_inches='tight')
plt.show()