import time

import numpy as np
import pandas as pd
import seaborn
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import windowing
from lstm_training_main import BiLSTMModel
from training import commons
import scenarios

# Constants
NUM_EPOCHS = 100
WINDOW_SIZE = 200
WINDOW_OVERLAP_SIZE = 0
BATCH_SIZE = 128
HIDDEN_SIZE = 256
INPUT_SIZE = 22
OUTPUT_SIZE = 5
LAYERS = 1

model_name = "LSTM"
plot_color = seaborn.color_palette("deep")[1]  # deep orange

# Constants
NUM_EPOCHS_ON_EACH_SCENARIO = 100
NUM_ITERATIONS_OVER_ALL_SCENARIOS = 3



def train_and_evaluate_model(base_model: BiLSTMModel, window_size, window_overlap, batch_size, hidden_size, layers):
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = base_model.to(device)

    # Start timer
    start_time = time.time()
    for iteration in range(0, NUM_ITERATIONS_OVER_ALL_SCENARIOS):
        # exclude main scenario
        for scenario in scenarios.scenarios[1:]:
            print(f"Iteration {iteration}, scenario {scenario.name}")
            train_loader, train_scalers = commons.load_data_with_platform_information(
                scenario.train_dataset_path, scenario.nodes_aux_path, scenario.links_aux_path,
                scenarios.input_columns_jobs_numerical, scenarios.input_columns_jobs_categorical,
                scenarios.output_columns_jobs_numerical, scenarios.output_columns_jobs_categorical,
                scenarios.nodes_columns_numerical, scenarios.nodes_columns_categorical,
                scenarios.links_columns_numerical, scenarios.links_columns_categorical,
                window_size, window_overlap, batch_size, do_shuffle=True)

            # Initialize the loss function, and optimizer

            criterion = nn.MSELoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # Learning rate should be lower than original
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

            # Training loop
            model.train()
            for epoch in range(NUM_EPOCHS_ON_EACH_SCENARIO):
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
                print(f'Epoch [{epoch + 1}/{NUM_EPOCHS_ON_EACH_SCENARIO}], Average Loss: {avg_loss}')
                scheduler.step(avg_loss)

    # Stop timer and print training summary
    end_time = time.time()
    total_time = end_time - start_time
    commons.print_training_summary(NUM_EPOCHS_ON_EACH_SCENARIO * NUM_ITERATIONS_OVER_ALL_SCENARIOS, window_size, window_overlap, batch_size, hidden_size, total_time)

    return model


if __name__ == '__main__':
    dir = "LSTM_4layer_512hs"
    base_model: BiLSTMModel = torch.load(f'generated-models/{dir}/lstm_base.pth')
    trained_model = train_and_evaluate_model(base_model, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, HIDDEN_SIZE, LAYERS)
    torch.save(trained_model, f'generated-models/{dir}/lstm_tuned_{NUM_EPOCHS_ON_EACH_SCENARIO}each_{NUM_ITERATIONS_OVER_ALL_SCENARIOS}iter.pth')