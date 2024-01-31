import time

import numpy as np
import pandas as pd
import seaborn
import torch
from torch import nn
from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import commons

# Constants
NUM_EPOCHS = 10
WINDOW_SIZE = 200
WINDOW_OVERLAP_SIZE = 100
BATCH_SIZE = 64
HIDDEN_LAYERS = 32
INPUT_SIZE = 4
OUTPUT_SIZE = 5
NHEADS = 2  # Ensure this is a divisor of HIDDEN_LAYERS
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

model_name = "Transformer"
plot_color = seaborn.color_palette("deep")[4]  # deep purple



class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModel, self).__init__()
        self.src_input_projection = nn.Linear(input_size, hidden_size)
        self.tgt_input_projection = nn.Linear(output_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_projection = nn.Linear(hidden_size, output_size)

    def forward(self, src, tgt):
        src = self.src_input_projection(src)
        tgt = self.tgt_input_projection(tgt)
        #print("src shape:", src.shape)  # Expected: [batch_size, seq_len_src, hidden_size]
        #print("tgt shape:", tgt.shape)  # Expected: [batch_size, seq_len_tgt, hidden_size]


        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        output = self.output_projection(output)
        return output

def shift_targets_for_decoder(targets):
    # Add a start token at the beginning and remove the last element
    start_token = torch.zeros((targets.shape[0], 1, targets.shape[2]), device=targets.device)
    decoder_input = torch.cat((start_token, targets[:, :-1, :]), dim=1)
    return decoder_input

def train_and_evaluate_model():
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Start timer
    start_time = time.time()

    # Load data
    # train_df = pd.read_csv('../../dataset_preparation/3rd-phase/train_dataset.csv')
    # test_df = pd.read_csv('../../dataset_preparation/3rd-phase/test_dataset.csv')
    train_df = pd.read_csv('../../../dataset_preparation/3rd-phase/train_dataset_small.csv')
    test_df = pd.read_csv('../../../dataset_preparation/3rd-phase/test_dataset_small.csv')

    input_columns = ['index', 'flops', 'input_files_size', 'output_files_size']
    output_columns = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']
    train_windows = commons.create_windows(train_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns, output_columns=output_columns)
    test_windows = commons.create_windows(test_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns, output_columns=output_columns)

    # Fit the scalers on the whole training dataset
    input_scaler, output_scaler = commons.create_and_fit_scalers(train_df, input_columns, output_columns)

    # Prepare datasets
    train_dataset = commons.scale_and_reshape_windows(train_windows, WINDOW_SIZE, input_scaler, output_scaler, input_columns, output_columns)
    test_dataset = commons.scale_and_reshape_windows(test_windows, WINDOW_SIZE, input_scaler, output_scaler, input_columns, output_columns)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model
    model = TransformerModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_LAYERS,
                             output_size=OUTPUT_SIZE, nhead=NHEADS,
                             num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # most accurate results so far for 0.001

    # Initialize the scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training Loop
    model.train()  # Set the model to training mode
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for input_window, actual_target_data_window in train_loader:
            input_window, actual_target_data_window = input_window.to(device), actual_target_data_window.to(device)
            # Shift the targets to create the input for the decoder
            shifted_decoder_input = shift_targets_for_decoder(actual_target_data_window)


            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_window, shifted_decoder_input)
            loss = criterion(outputs, actual_target_data_window)
            # Backward pass
            loss.backward()

            # It helps, nan otherwise
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Average Loss: {avg_loss}')
        scheduler.step(avg_loss)

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
        # Initialize the decoder input for the first step

        for input_window, actual_target_data_window in test_loader:
            input_window = input_window.to(device)

            # Forward pass through the model
            decoder_input = torch.zeros((input_window.shape[0], WINDOW_SIZE, OUTPUT_SIZE), device=device)
            output = model(input_window, decoder_input)

            # Store predictions and actual values for further metrics calculations
            predictions.extend(output.cpu().numpy())
            actual_values.extend(actual_target_data_window.cpu().numpy())

    # Convert lists of arrays to single numpy arrays
    predictions_array = np.vstack(predictions)
    actual_values_array = np.vstack(actual_values)

    # Calculate metrics for each output parameter and show them
    commons.calculate_and_show_metrics(output_columns, predictions_array, actual_values_array)

    # Denormalize and plot results for each parameter
    commons.denorm_and_plot_predicted_actual(output_columns, output_scaler, predictions_array, actual_values_array, model_name, purpose="training")

    return model


if __name__ == '__main__':
    model = train_and_evaluate_model()
    torch.save(model.state_dict(), '../generated-models/transformer_weights.pth')
    torch.save(model, '../generated-models/transformer-encoder-100ep-64ba-256hl-4he-4en/transformer.pth')
