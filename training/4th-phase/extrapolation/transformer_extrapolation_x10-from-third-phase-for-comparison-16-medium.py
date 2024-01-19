import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import windowing
import plotting
from initial_training.third_phase_transformer import TransformerModel, model_name, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, plot_color

# Load dataframes
test_df = pd.read_csv('../../../dataset-preparation/4th-phase/extrapolation/16-kit_desy-cache-moved-to-desy-more-resources-in-desy-less-resources-in-kit-faster-links-extrapolation/extrapolation_dataset.csv', sep=";")

# Load jobs
input_columns_jobs = ['index', 'flops', 'input_files_size', 'output_files_size']
output_columns_jobs = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']

test_windows = windowing.create_windows_jobs(test_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns_jobs,
                                             output_columns=output_columns_jobs)


# Fit the scalers on the whole training dataset
input_jobs_scaler = windowing.create_and_fit_scaler(test_df, input_columns_jobs)
output_jobs_scaler = windowing.create_and_fit_scaler(test_df, output_columns_jobs)

# Prepare datasets
test_dataset = windowing.scale_and_reshape_jobs_windows(test_windows, WINDOW_SIZE, input_jobs_scaler, output_jobs_scaler, input_columns_jobs, output_columns_jobs)


test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)




def apply_model_to_data():
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = torch.load('../../../trained-models/3rd-phase/transformer.pth')

    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Evaluate on test data
    predictions = []
    actual_values = []

    with torch.no_grad():
        for inputs, targets in test_loader:
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
    plotting.calculate_and_show_metrics(output_columns_jobs, predictions_array, actual_values_array)

    # Denormalize and plot results for each parameter
    plotting.denorm_and_plot_predicted_actual(output_columns_jobs, output_jobs_scaler, predictions_array, actual_values_array, model_name, color_name=plot_color, purpose="3rd-phase-model-16-medium_extrapolation_x10")

    # Create KDE plot
    plotting.plot_kde(output_columns_jobs, predictions_array, actual_values_array, model_name, color_name=plot_color, purpose="3rd-phase-model-16-medium_extrapolation_x10")


if __name__ == '__main__':
    apply_model_to_data()
