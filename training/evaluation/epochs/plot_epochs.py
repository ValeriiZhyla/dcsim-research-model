import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Define the file paths
file_paths = [
    "data/gru_1.csv", "data/gru_2.csv", "data/gru_3.csv", "data/gru_4.csv", "data/gru_5.csv",
    "data/lstm_1.csv", "data/lstm_2.csv", "data/lstm_3.csv", "data/lstm_4.csv", "data/lstm_5.csv",
    "data/transformer_1.csv", "data/transformer_2.csv", "data/transformer_3.csv",  "data/transformer_4.csv", "data/transformer_5.csv",
]

# Colors for each type
colors = {'gru': 'blue', 'lstm': 'orange', 'transformer': 'violet'}

# Creating a DataFrame to hold all the data
df_all = pd.DataFrame()

for file_path in file_paths:
    # Determine the model type based on the file name
    if 'gru' in file_path:
        model_type = 'gru'
    elif 'lstm' in file_path:
        model_type = 'lstm'
    else:
        model_type = 'transformer'

    # Load the data
    temp_df = pd.read_csv(file_path, header=None)
    temp_df['Epoch'] = temp_df.index
    temp_df['Model'] = model_type

    # Append to the overall DataFrame
    df_all = pd.concat([df_all, temp_df], axis=0)

# Rename columns for clarity
df_all.columns = ['Value', 'Epoch', 'Model']

# Plot
plt.figure(figsize=(10, 6))
seaborn.lineplot(data=df_all, x='Epoch', y='Value', hue='Model', style='Model', palette=colors, markers=True, dashes=False)
plt.title('Model Training Comparison', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend(title='Model Type', fontsize='large')
plt.savefig('plots/loss_vs_epochs.png', dpi=200, bbox_inches='tight')
plt.show()
