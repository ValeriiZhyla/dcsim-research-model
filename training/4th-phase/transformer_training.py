import time

import numpy as np
import pandas as pd
import seaborn
import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader

import plotting
import windowing

# Constants
NUM_EPOCHS = 100
WINDOW_SIZE = 500
WINDOW_OVERLAP_SIZE = 250
BATCH_SIZE = 32
HIDDEN_LAYERS = 15  # 8 hidden layers produce NaN loss, 5 Produces good results, 10 produces very good results
INPUT_SIZE = 4
OUTPUT_SIZE = 5
NHEADS = 1  # Ensure this is a divisor of HIDDEN_LAYERS
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

model_name = "Transformer"
plot_color = seaborn.color_palette("deep")[4]  # deep purple



class TransformerModelWithTwoAuxEncoders(nn.Module):
    def __init__(self, jobs_src_input_size, nodes_aux_input_size, links_input_size, hidden_size, output_size, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModelWithTwoAuxEncoders, self).__init__()
        # Linear layers to project source, two auxiliary, and target features to hidden dimension
        self.jobs_src_input_projection = nn.Linear(jobs_src_input_size, hidden_size)
        self.nodes_aux_input_projection = nn.Linear(nodes_aux_input_size, hidden_size)
        self.links_aux_input_projection = nn.Linear(links_input_size, hidden_size)
        self.tgt_input_projection = nn.Linear(output_size, hidden_size)

        # Separate encoders for source and two auxiliary sequences
        self.jobs_src_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.jobs_src_encoder = nn.TransformerEncoder(self.jobs_src_encoder_layer, num_layers=num_encoder_layers)

        self.nodes_aux_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.nodes_aux_encoder = nn.TransformerEncoder(self.nodes_aux_encoder_layer, num_layers=num_encoder_layers)

        self.links_aux_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.links_aux_encoder = nn.TransformerEncoder(self.links_aux_encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder layer
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead)
        self.decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_decoder_layers)

        # Linear layer to project from hidden dimension to output size
        self.output_projection = nn.Linear(hidden_size, output_size)

    def forward(self, jobs, nodes, links, tgt):
        # Project to hidden size
        jobs = self.jobs_src_input_projection(jobs)
        nodes = self.nodes_aux_input_projection(nodes)
        links = self.links_aux_input_projection(links)
        tgt = self.tgt_input_projection(tgt)

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

def train_and_evaluate_model():
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Start timer
    start_time = time.time()

    # Load data
    train_df = pd.read_csv('../../dataset-preparation/3rd-phase/train_dataset.csv')
    test_df = pd.read_csv('../../dataset-preparation/3rd-phase/test_dataset.csv')
    #train_df = pd.read_csv('../../dataset-preparation/3rd-phase/train_dataset_small.csv')
    #test_df = pd.read_csv('../../dataset-preparation/3rd-phase/test_dataset_small.csv')

    input_columns = ['index', 'flops', 'input_files_size', 'output_files_size']
    output_columns = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']
    train_windows = windowing.create_windows(train_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns, output_columns=output_columns)
    test_windows = windowing.create_windows(test_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns, output_columns=output_columns)

    # Fit the scalers on the whole training dataset
    input_scaler, output_scaler = windowing.create_and_fit_scalers(train_df, input_columns, output_columns)

    # Prepare datasets
    train_dataset = windowing.scale_and_reshape_windows(train_windows, WINDOW_SIZE, input_scaler, output_scaler, input_columns, output_columns)
    test_dataset = windowing.scale_and_reshape_windows(test_windows, WINDOW_SIZE, input_scaler, output_scaler, input_columns, output_columns)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model
    model = TransformerModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_LAYERS,
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
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, targets)
            loss = criterion(outputs, targets)
            # Backward pass
            loss.backward()

            # It helps, nan otherwise
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

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
    plotting.calculate_and_show_metrics(output_columns, predictions_array, actual_values_array)

    # Denormalize and plot results for each parameter
    plotting.denorm_and_plot_predicted_actual(output_columns, output_scaler, predictions_array, actual_values_array, model_name, purpose="training")

    return model


if __name__ == '__main__':
    model = train_and_evaluate_model()
    torch.save(model.state_dict(), 'generated-models/transformer_weights.pth')
    torch.save(model, 'generated-models/transformer.pth')
