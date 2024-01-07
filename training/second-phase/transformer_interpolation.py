import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import commons
from transformer_training import TransformerModel, model_name, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, plot_color

train_df = pd.read_csv('../../dataset-preparation/second-phase/test_dataset.csv')
input_columns = ['index', 'flops', 'input_files_size', 'output_files_size']
output_columns = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']
apply_data_windows = commons.create_windows(train_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns, output_columns=output_columns)

# Fit the scalers on the whole training dataset
input_scaler, output_scaler = commons.create_and_fit_scalers(train_df, input_columns, output_columns)

transformed_dataset = commons.scale_and_reshape_windows(apply_data_windows, WINDOW_SIZE, input_scaler, output_scaler, input_columns, output_columns)
loader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE, shuffle=False)


def apply_model_to_data():
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = torch.load('../../trained-models/second-phase/transformer.pth')
    # model = torch.load('generated-models/transformer.pth')

    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Evaluate on test data
    predictions = []
    actual_values = []

    with torch.no_grad():
        for inputs, targets in loader:
            # Move data to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Make a prediction
            outputs = model(inputs, targets)

            # Store predictions and actual values for further metrics calculations
            predictions.extend(outputs.cpu().numpy())
            actual_values.extend(targets.cpu().numpy())

    # Convert lists of arrays to single numpy arrays
    predictions_array = np.vstack(predictions)
    actual_values_array = np.vstack(actual_values)

    # Calculate metrics for each output parameter and show them
    commons.calculate_and_show_metrics(output_columns, predictions_array, actual_values_array)

    # Denormalize and plot results for each parameter
    commons.denorm_and_plot(output_columns, output_scaler, predictions_array, actual_values_array, model_name, color_name=plot_color, purpose="interpolation")


if __name__ == '__main__':
    apply_model_to_data()
