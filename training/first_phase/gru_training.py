import time

import numpy as np
import seaborn
import torch
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset
import commons

# Constants
NUM_EPOCHS = 1000
WINDOW_SIZE = 500
WINDOW_OVERLAP_SIZE = 250
BATCH_SIZE = 128
HIDDEN_LAYERS = 50
INPUT_SIZE = 4
OUTPUT_SIZE = 5

model_name = "GRU"

# Define the GRU Model
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    def forward(self, x):
        # GRU layer
        out, _ = self.gru(x)  # out shape: [batch_size, sequence_length, hidden_size * 2]

        # Apply the linear layer to each time step
        out = self.fc(out)  # out shape: [batch_size, sequence_length, output_size]
        return out


def train_and_evaluate_model():
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Start timer
    start_time = time.time()

    # Load data
    #train_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/train_dataset.csv')
    #test_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/test_dataset.csv')
    train_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/train_dataset_small.csv')
    test_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/test_dataset_small.csv')

    # Transform dataframes into overlapping windows
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

    # Initialize the model, loss function, and optimizer
    model = BiGRUModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_LAYERS, output_size=OUTPUT_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Average Loss: {avg_loss}')

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
            outputs = model(inputs)

            # Store predictions and actual values for further metrics calculations
            predictions.extend(outputs.cpu().numpy())
            actual_values.extend(targets.cpu().numpy())

    # Convert lists of arrays to single numpy arrays
    predictions_array = np.vstack(predictions)
    actual_values_array = np.vstack(actual_values)

    # Calculate metrics for each output parameter and show them
    commons.calculate_and_show_metrics(output_columns, predictions_array, actual_values_array)

    # Denormalize and plot results for each parameter
    commons.denorm_and_plot(output_columns, output_scaler, predictions_array, actual_values_array, model_name)
    return model


if __name__ == '__main__':
    model = train_and_evaluate_model()
    torch.save(model.state_dict(), 'gru_weights.pth')
    torch.save(model, 'gru.pth')




