import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import windowing
import plotting
from transformer_training import TransformerModelWithTwoAuxEncoders, CombinedDataset, model_name, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, plot_color

# Load dataframes
test_df = pd.read_csv('../../dataset-preparation/4th-phase/main-dataset/extrapolation_dataset_x20_extended.csv')
nodes_aux_df = pd.read_csv('../../dataset-preparation/4th-phase/main-dataset/nodes_aux.csv')
links_aux_df = pd.read_csv('../../dataset-preparation/4th-phase/main-dataset/links_aux.csv')

# Load jobs
input_columns_jobs = ['index', 'flops', 'input_files_size', 'output_files_size', 'dataset_node_index']
output_columns_jobs = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time', 'machine_index']

test_windows = windowing.create_windows_jobs(test_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns_jobs,
                                             output_columns=output_columns_jobs)

# Load aux data
nodes_columns = ['index', 'type_index', 'speed_mf', 'cores', 'ram_gib', 'disk_tib', 'disk_read_bw_mbps', 'disk_write_bw_mbps', 'in_cluster']
links_columns = ['link_index', 'src_node_index', 'dst_node_index', 'bandwidth_mbps', 'latency_us', 'is_fatpipe']
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


def apply_model_to_data():
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #model = torch.load('../../trained-models/4th-phase/transformer.pth')
    model = torch.load('generated-models/transformer.pth')

    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Evaluate on test data
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
    plotting.calculate_and_show_metrics(output_columns_jobs, predictions_array, actual_values_array)

    # Denormalize and plot results for each parameter
    plotting.denorm_and_plot_predicted_actual(output_columns_jobs, output_jobs_scaler, predictions_array, actual_values_array, model_name, color_name=plot_color, purpose="extrapolation_x20")

    # Create KDE plot
    plotting.plot_kde(output_columns_jobs, predictions_array, actual_values_array, model_name, color_name=plot_color, purpose="extrapolation_x20")


if __name__ == '__main__':
    apply_model_to_data()
