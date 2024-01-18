import os
import sys
import time

import numpy as np
import pandas as pd
import seaborn
import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader, Dataset

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotting
import windowing

# Constants
NUM_EPOCHS = 30
WINDOW_SIZE = 500
WINDOW_OVERLAP_SIZE = 250
BATCH_SIZE = 32
HIDDEN_LAYERS = 20  # 8 hidden layers produce NaN loss, 5 Produces good results, 10 produces very good results
INPUT_SIZE = 5
OUTPUT_SIZE = 6
NHEADS = 1  # Ensure this is a divisor of HIDDEN_LAYERS
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

model_name = "Transformer"
plot_color = seaborn.color_palette("deep")[4]  # deep purple


class TransformerModelWithTwoAuxEncodersAndNormalisation(nn.Module):
    def __init__(self, jobs_src_input_size, nodes_aux_input_size, links_input_size, hidden_size, output_size, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModelWithTwoAuxEncodersAndNormalisation, self).__init__()
        # Linear layers and BatchNorm1d layers for source, two auxiliary, and target features to hidden dimension
        self.jobs_src_input_projection = nn.Linear(jobs_src_input_size, hidden_size)
        self.jobs_src_bn = nn.BatchNorm1d(hidden_size)
        self.nodes_aux_input_projection = nn.Linear(nodes_aux_input_size, hidden_size)
        self.nodes_aux_bn = nn.BatchNorm1d(hidden_size)
        self.links_aux_input_projection = nn.Linear(links_input_size, hidden_size)
        self.links_aux_bn = nn.BatchNorm1d(hidden_size)
        self.tgt_input_projection = nn.Linear(output_size, hidden_size)
        self.tgt_bn = nn.BatchNorm1d(hidden_size)

        # Separate encoders for source and two auxiliary sequences
        self.jobs_src_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.jobs_src_encoder = nn.TransformerEncoder(self.jobs_src_encoder_layer, num_layers=num_encoder_layers)

        self.nodes_aux_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.nodes_aux_encoder = nn.TransformerEncoder(self.nodes_aux_encoder_layer, num_layers=num_encoder_layers)

        self.links_aux_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.links_aux_encoder = nn.TransformerEncoder(self.links_aux_encoder_layer, num_layers=num_encoder_layers)

        # Combination layer
        self.combination_layer = nn.Linear(hidden_size * 3, hidden_size)


        # Transformer decoder layer
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead)
        self.decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_decoder_layers)

        # Linear layer to project from hidden dimension to output size
        self.output_projection = nn.Linear(hidden_size, output_size)


    def forward(self, jobs, nodes, links, tgt):
        # Project to hidden size and apply BatchNorm
        jobs = self.jobs_src_input_projection(jobs).transpose(1, 2)
        jobs = self.jobs_src_bn(jobs).transpose(1, 2)
        nodes = self.nodes_aux_input_projection(nodes).transpose(1, 2)
        nodes = self.nodes_aux_bn(nodes).transpose(1, 2)
        links = self.links_aux_input_projection(links).transpose(1, 2)
        links = self.links_aux_bn(links).transpose(1, 2)
        tgt = self.tgt_input_projection(tgt).transpose(1, 2)
        tgt = self.tgt_bn(tgt).transpose(1, 2)

        # Encode source and auxiliary sequences separately
        encoded_jobs_src = self.jobs_src_encoder(jobs)
        encoded_nodes_aux = self.nodes_aux_encoder(nodes)
        encoded_links_aux = self.links_aux_encoder(links)

        # Combine encoded source and both auxiliary sequences
        # Combine the encoded representations
        combined = torch.cat([encoded_jobs_src, encoded_nodes_aux, encoded_links_aux], dim=-1)
        combined = self.combination_layer(combined)  # Combination layer (e.g., linear)

        # Use combined memory for the decoder, and target for queries
        output = self.decoder(tgt, combined)

        # Project output to target size
        return self.output_projection(output)


class CombinedDataset(Dataset):
    def __init__(self, train_dataset, nodes_aux_data, links_aux_data):
        self.train_dataset = train_dataset
        self.nodes_aux_data = nodes_aux_data
        self.links_aux_data = links_aux_data

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        train_data, train_target = self.train_dataset[idx]
        # Since nodes_aux_data and links_aux_data are single windows, use them as is
        nodes_aux_data = self.nodes_aux_data.squeeze(0)  # Remove batch dimension if present
        links_aux_data = self.links_aux_data.squeeze(0)  # Remove batch dimension if present
        return train_data, nodes_aux_data, links_aux_data, train_target

def train_and_evaluate_model():
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Start timer
    start_time = time.time()

    # Load dataframes
    train_df = pd.read_csv('../../../dataset-preparation/4th-phase/0-main-dataset/train_dataset.csv', sep=";")
    test_df = pd.read_csv('../../../dataset-preparation/4th-phase/0-main-dataset/test_dataset.csv', sep=";")
    nodes_aux_df = pd.read_csv('../../../dataset-preparation/4th-phase/0-main-dataset/nodes_aux.csv', sep=";")
    links_aux_df = pd.read_csv('../../../dataset-preparation/4th-phase/0-main-dataset/links_aux.csv', sep=";")

    # Load jobs
    input_columns_jobs = ['index', 'flops', 'input_files_size', 'output_files_size', 'dataset_node_index']
    output_columns_jobs = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time', 'machine_index']

    train_windows = windowing.create_windows_jobs(train_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns_jobs,
                                                  output_columns=output_columns_jobs)
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
    input_jobs_scaler = windowing.create_and_fit_scaler(train_df, input_columns_jobs)
    output_jobs_scaler = windowing.create_and_fit_scaler(train_df, output_columns_jobs)
    nodes_scaler = windowing.create_and_fit_scaler(nodes_aux_df, nodes_columns)
    links_scaler = windowing.create_and_fit_scaler(links_aux_df, links_columns)

    # Prepare datasets
    train_dataset = windowing.scale_and_reshape_jobs_windows(train_windows, WINDOW_SIZE, input_jobs_scaler, output_jobs_scaler, input_columns_jobs, output_columns_jobs)
    test_dataset = windowing.scale_and_reshape_jobs_windows(test_windows, WINDOW_SIZE, input_jobs_scaler, output_jobs_scaler, input_columns_jobs, output_columns_jobs)
    nodes_aux_dataset = windowing.scale_and_reshape_aux_windows(nodes_aux_windows, WINDOW_SIZE, nodes_scaler, nodes_columns)
    links_aux_dataset = windowing.scale_and_reshape_aux_windows(links_aux_windows, WINDOW_SIZE, links_scaler, links_columns)

    train_dataset_jobs_with_aux = CombinedDataset(train_dataset, nodes_aux_dataset, links_aux_dataset)
    test_dataset_jobs_with_aux = CombinedDataset(test_dataset, nodes_aux_dataset, links_aux_dataset)

    train_loader = DataLoader(train_dataset_jobs_with_aux, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset_jobs_with_aux, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model
    model = TransformerModelWithTwoAuxEncodersAndNormalisation(jobs_src_input_size=INPUT_SIZE, nodes_aux_input_size=len(nodes_columns), links_input_size=len(links_columns),
                                                               hidden_size=HIDDEN_LAYERS,
                                                               output_size=OUTPUT_SIZE, nhead=NHEADS,
                                                               num_encoder_layers=NUM_ENCODER_LAYERS,
                                                               num_decoder_layers=NUM_DECODER_LAYERS).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)  # most accurate results so far for 0.001
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training Loop
    model.train()  # Set the model to training mode
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch in train_loader:
            # Unpack the batch data
            train_data, nodes_aux_data, links_aux_data, train_target = batch

            # Move data to the device
            train_data = train_data.to(device)
            nodes_aux_data = nodes_aux_data.to(device)
            links_aux_data = links_aux_data.to(device)
            train_target = train_target.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(train_data, nodes_aux_data, links_aux_data, train_target)
            loss = criterion(outputs, train_target)

            # Backward pass
            loss.backward()
            # It helps, nan otherwise
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Average Loss: {avg_loss}')
    # loss converges to +- 0.092
    # some starts to +- 0.042

    # Stop timer
    end_time = time.time()
    total_time = end_time - start_time
    # Print training summary
    print("=================================")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Window overlap: {WINDOW_OVERLAP_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Hidden layers: {HIDDEN_LAYERS}")
    print(f"Total time for training: {total_time:.2f} seconds")
    print("=================================")

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
    plotting.calculate_and_show_metrics(output_columns_jobs, predictions_array, actual_values_array)

    # Denormalize and plot results for each parameter
    plotting.denorm_and_plot_predicted_actual(output_columns_jobs, output_jobs_scaler, predictions_array, actual_values_array, model_name, purpose="training")

    return model


if __name__ == '__main__':
    model = train_and_evaluate_model()
    torch.save(model.state_dict(), '../../../trained-models/4th-phase/initial_model_batch_normalisation/transformer_with_batch_normalisation_weights.pth')
    torch.save(model, '../../../trained-models/4th-phase/initial_model_batch_normalisation/transformer_with_batch_normalisation.pth')
