import time

import numpy as np
import pandas as pd
import seaborn
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import windowing
from training import commons
import scenarios

# Constants
NUM_EPOCHS = 200
WINDOW_SIZE = 200
WINDOW_OVERLAP_SIZE = 100
BATCH_SIZE = 128
HIDDEN_SIZE = 125
INPUT_SIZE = 6
OUTPUT_SIZE = 5
LAYERS = 1

model_name = "LSTM"
plot_color = seaborn.color_palette("deep")[1]  # deep orange

# Constants
NUM_EPOCHS_ON_EACH_SCENARIO = 100
NUM_ITERATIONS_OVER_ALL_SCENARIOS = 10

# Define the LSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    def forward(self, x):
        # LSTM layer
        out, _ = self.lstm(x)  # out shape: [batch_size, sequence_length, hidden_size * 2]

        # Apply the linear layer to each time step
        out = self.fc(out)  # out shape: [batch_size, sequence_length, output_size]
        return out


def train_and_evaluate_model(num_epochs, window_size, window_overlap, batch_size, hidden_size, layers):
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Start timer
    start_time = time.time()
    scenario = scenarios.main_scenario

    # Load data and scalers
    train_df = pd.read_csv(scenario.train_dataset_path, delimiter=';')

    # Fit the scalers on the whole training dataset for numerical columns
    train_scalers, train_df_numerical_scaled_categorical_original = commons.df_fit_transform_and_get_scalers(train_df, scenarios.input_columns_jobs_numerical + scenarios.output_columns_jobs_numerical)

    # windows with scaled numerical and original categorical data
    train_windows = windowing.create_windows(train_df_numerical_scaled_categorical_original, window_size=window_size, overlap_size=window_overlap, input_columns=scenarios.input_columns_jobs_numerical + scenarios.input_columns_jobs_categorical,
                                             output_columns=scenarios.output_columns_jobs_numerical + scenarios.output_columns_jobs_categorical)

    # df with nodes and df with links
    nodes_df, links_df = commons.load_nodes_and_links_padded(scenario.nodes_aux_path, scenario.links_aux_path, scenarios.nodes_columns_numerical + scenarios.nodes_columns_categorical,
                                                             scenarios.links_columns_numerical + scenarios.links_columns_categorical, window_size)
    # df with nodes and df with links as nparrays
    nodes_df_np = nodes_df.to_numpy()
    links_df_np = links_df.to_numpy()

    # add the aux data to each input window
    train_windows_with_aux = [(np.hstack((window[0], nodes_df_np, links_df_np)), window[1]) for window in train_windows]

    # data should include categorical and aux data
    train_dataset = commons.create_tensor_dataset(train_windows_with_aux)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    # Initialize the model, loss function, and optimizer
    model = BiLSTMModel(input_size=INPUT_SIZE, hidden_size=hidden_size, output_size=OUTPUT_SIZE, num_layers=layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs_jobs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss}')
        scheduler.step(avg_loss)

    # Stop timer and print training summary
    end_time = time.time()
    total_time = end_time - start_time
    commons.print_training_summary(num_epochs, window_size, window_overlap, batch_size, hidden_size, total_time)

    return model


if __name__ == '__main__':
    model = train_and_evaluate_model(NUM_EPOCHS, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, HIDDEN_SIZE, LAYERS)
    torch.save(model.state_dict(), 'generated-models/default/lstm_weights.pth')
    torch.save(model, 'generated-models/default/lstm.pth')
