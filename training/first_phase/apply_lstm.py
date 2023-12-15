import pandas as pd
import torch
from torch.utils.data import DataLoader

import commons

WINDOW_SIZE = 500
WINDOW_OVERLAP_SIZE = 250
BATCH_SIZE = 128

model = torch.load('lstm.pth')
model.eval()  # Set the model to evaluation mode

train_df = pd.read_csv('sequence-to-predict.csv')
input_columns = ['flops', 'input_files_size', 'output_files_size']
output_columns = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']
apply_data_windows = commons.create_windows(train_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns, output_columns=output_columns)

# Fit the scalers on the whole training dataset
input_scaler, output_scaler = commons.create_and_fit_scalers(train_df, input_columns, output_columns)

transformed_dataset = commons.process_windows(apply_data_windows, WINDOW_SIZE, input_scaler, output_scaler, input_columns, output_columns)
loader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.to('cpu')  # or 'cuda:0' if using GPU

# Make predictions
with torch.no_grad():
    predictions = []
    for inputs, targets in loader:
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())

print("XXXX")