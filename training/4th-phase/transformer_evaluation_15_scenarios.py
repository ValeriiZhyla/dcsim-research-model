import time

import numpy as np
import pandas as pd
import seaborn
import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader, Dataset
from transformer_initial_training import TransformerModelWithTwoAuxEncoders, CombinedDataset, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, HIDDEN_LAYERS, INPUT_SIZE, OUTPUT_SIZE, \
    NHEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, model_name, plot_color
from transformer_fine_tuning_15_scenarios import Scenario, scenarios, input_columns_jobs, output_columns_jobs, nodes_columns, links_columns

import plotting
import windowing

def fine_tune_model(model: TransformerModelWithTwoAuxEncoders):
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Start timer
    start_time = time.time()

    for scenario in scenarios:
        # Load dataframes
        train_df, test_df, nodes_aux_df, links_aux_df = scenario.read_all_datasets()

        # Load jobs
        test_windows = windowing.create_windows_jobs(test_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns_jobs,
                                                     output_columns=output_columns_jobs)

        # Load aux data
        nodes_aux_df = nodes_aux_df[nodes_columns]
        links_aux_df = links_aux_df[links_columns]
        nodes_aux_windows = windowing.pad_window_aux(nodes_aux_df, window_size=WINDOW_SIZE, columns=nodes_columns)
        links_aux_windows = windowing.pad_window_aux(links_aux_df, window_size=WINDOW_SIZE, columns=links_columns)

        # Fit the scalers on the whole training dataset
        input_jobs_scaler = windowing.create_and_fit_scaler(test_df, input_columns_jobs)
        output_jobs_scaler = windowing.create_and_fit_scaler(test_df, output_columns_jobs)
        nodes_scaler = windowing.create_and_fit_scaler(nodes_aux_df, nodes_columns)
        links_scaler = windowing.create_and_fit_scaler(links_aux_df, links_columns)

        # Prepare datasets
        test_dataset = windowing.scale_and_reshape_jobs_windows(test_windows, WINDOW_SIZE, input_jobs_scaler, output_jobs_scaler, input_columns_jobs, output_columns_jobs)
        nodes_aux_dataset = windowing.scale_and_reshape_aux_windows(nodes_aux_windows, WINDOW_SIZE, nodes_scaler, nodes_columns)
        links_aux_dataset = windowing.scale_and_reshape_aux_windows(links_aux_windows, WINDOW_SIZE, links_scaler, links_columns)

        test_dataset_jobs_with_aux = CombinedDataset(test_dataset, nodes_aux_dataset, links_aux_dataset)

        test_loader = DataLoader(test_dataset_jobs_with_aux, batch_size=BATCH_SIZE, shuffle=False)

        # Evaluate the model with test data
        model.eval()
        predictions = []
        actual_values = []

        with torch.no_grad():
            for batch in test_loader:
                # Unpack the batch data
                train_data, nodes_aux_data, links_aux_data, train_target = batch

                # Move data to the device
                train_data = train_data.to(device)
                nodes_aux_data = nodes_aux_data.to(device)
                links_aux_data = links_aux_data.to(device)
                train_target = train_target.to(device)

                # Make a prediction
                outputs = model(train_data, nodes_aux_data, links_aux_data, train_target)

                # Store predictions and actual values for further metrics calculations
                predictions.extend(outputs.cpu().numpy())
                actual_values.extend(train_target.cpu().numpy())

        # Convert lists of arrays to single numpy arrays
        predictions_array = np.vstack(predictions)
        actual_values_array = np.vstack(actual_values)

        # Calculate metrics for each output parameter and show them
        plotting.calculate_and_show_metrics(output_columns_jobs, predictions_array, actual_values_array, scenario_name=scenario.name)

        # Denormalize and plot results for each parameter
        plotting.denorm_and_plot_predicted_actual(output_columns_jobs, output_jobs_scaler, predictions_array, actual_values_array, model_name, purpose=scenario.name + " interpolation")

if __name__ == '__main__':
    model = torch.load('generated-models/transformer_tuned_on_15_scenarios.pth')
    fine_tune_model(model)
