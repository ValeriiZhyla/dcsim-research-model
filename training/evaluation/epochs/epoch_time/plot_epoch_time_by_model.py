# Re-importing necessary libraries and redefining the plotting code after the reset.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

# Define your data
data = {
    'sequences': [1, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500],
    'GRU': [0.0139, 0.0162, 0.0309, 0.0442, 0.0571, 0.0690, 0.1149, 0.1813, 0.2325, 0.3228, 0.4568, 0.5994, 0.7157],
    'LSTM': [0.0142, 0.0197, 0.0332, 0.0469, 0.0633, 0.0792, 0.1209, 0.1832, 0.2751, 0.3748, 0.5131, 0.6613, 0.8556],
    'Transformer': [0.0190, 0.0262, 0.0501, 0.0666, 0.0896, 0.1109, 0.1655, 0.2510, 0.4030, 0.4952, 0.7100, 0.9303, 1.1322]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame to make it suitable for seaborn lineplot
df_melted = df.melt(id_vars=["sequences"], var_name="Model", value_name="Epoch Duration")

# Colors for each type
colors = {'GRU': 'blue', 'LSTM': 'orange', 'Transformer': 'violet'}

# Plot
plt.figure(figsize=(10, 6))
seaborn.lineplot(data=df_melted, x='sequences', y='Epoch Duration', hue='Model', palette=colors, marker='o')
plt.title('Epoch Duration of Different Models Across Sequences of Varying Lengths', fontsize=16)
plt.xlabel('Sequences of Each Length', fontsize=16)
plt.ylabel('Epoch Duration (s)', fontsize=16)
plt.legend(title='Model Type', fontsize='large')
plt.grid(True)
plt.savefig('plots/epoch_duration_vs_sequences.png', dpi=200, bbox_inches='tight')
plt.show()