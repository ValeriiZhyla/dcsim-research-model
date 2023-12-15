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
HIDDEN_LAYERS = 30


# Define the LSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    def forward(self, x):
        # LSTM layer
        out, _ = self.lstm(x)  # out shape: [batch_size, sequence_length, hidden_size * 2]

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
    train_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/train_dataset.csv')
    test_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/test_dataset.csv')
    #train_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/train_dataset_small.csv')
    #test_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/test_dataset_small.csv')

    # Transform dataframes into overlapping windows
    input_columns = ['flops', 'input_files_size', 'output_files_size']
    output_columns = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']
    train_windows = commons.create_windows(train_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns, output_columns=output_columns)
    test_windows = commons.create_windows(test_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns, output_columns=output_columns)

    # Fit the scalers on the whole training dataset
    input_scaler, output_scaler = commons.create_and_fit_scalers(train_df, input_columns, output_columns)

    # Prepare datasets
    train_dataset = commons.process_windows(train_windows, WINDOW_SIZE, input_scaler, output_scaler, input_columns, output_columns)
    test_dataset = commons.process_windows(test_windows, WINDOW_SIZE, input_scaler, output_scaler, input_columns, output_columns)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = BiLSTMModel(input_size=3, hidden_size=HIDDEN_LAYERS, output_size=5).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for windows, targets in train_loader:
            windows, targets = windows.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(windows)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item()}')

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
    test_loss = 0
    predictions = []
    actual_values = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Store predictions and actual values for further metrics calculations
            predictions.extend(outputs.cpu().numpy())
            actual_values.extend(targets.cpu().numpy())

    # Convert lists of arrays to single numpy arrays
    predictions_array = np.vstack(predictions)
    actual_values_array = np.vstack(actual_values)

    # Calculate metrics for each output parameter
    mse_per_param = []
    mae_per_param = []
    rmse_per_param = []
    r_squared_per_param = []

    for i in range(len(output_columns)):
        # Extract predictions and actual values for the current parameter
        actual_param = actual_values_array[:, i]
        predicted_param = predictions_array[:, i]

        # Calculate metrics for this parameter
        mse = mean_squared_error(actual_param, predicted_param)
        mae = mean_absolute_error(actual_param, predicted_param)
        rmse = np.sqrt(mse)
        r_squared = r2_score(actual_param, predicted_param)

        # Store the metrics
        mse_per_param.append(mse)
        mae_per_param.append(mae)
        rmse_per_param.append(rmse)
        r_squared_per_param.append(r_squared)

    # Print metrics for each parameter
    for i in range(len(output_columns)):
        print(f"Parameter [{output_columns[i]}]:")
        print(f"  Mean Squared Error (MSE):         {str(round(mse_per_param[i], 7)).rjust(15)}")
        print(f"  Root Mean Squared Error (RMSE):   {str(round(rmse_per_param[i], 7)).rjust(15)}")
        print(f"  Mean Absolute Error (MAE):        {str(round(mae_per_param[i], 7)).rjust(15)}")
        print(f"  R-squared (R^2):                  {str(round(r_squared_per_param[i], 7)).rjust(15)}")
        print("=================================")

    # Scale back to original values for readable plots
    denorm_predictions_array = output_scaler.inverse_transform(predictions_array)
    denorm_actual_values_array = output_scaler.inverse_transform(actual_values_array)

    # Create a DataFrame for plotting
    for i, label in enumerate(output_columns):
        # Create a DataFrame for plotting
        data_for_plot = pd.DataFrame({
            'Actual Values': denorm_actual_values_array[:, i],
            'Predictions': denorm_predictions_array[:, i]
        })

        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        seaborn.scatterplot(data=data_for_plot, x='Actual Values', y='Predictions', alpha=0.3)
        plt.title(f'{label}')
        plt.show()
    return model


if __name__ == '__main__':
    model = train_and_evaluate_model()
    torch.save(model, 'lstm.pth')




