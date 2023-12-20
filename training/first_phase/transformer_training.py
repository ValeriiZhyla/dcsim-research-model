import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import commons

# Constants
NUM_EPOCHS = 1000
WINDOW_SIZE = 100
WINDOW_OVERLAP_SIZE = 50
BATCH_SIZE = 64
HIDDEN_LAYERS = 64
INPUT_SIZE = 4
OUTPUT_SIZE = 5
NHEADS = 1  # Ensure this is a divisor of HIDDEN_LAYERS
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

model_name = "Transformer"


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModel, self).__init__()
        self.embedding_src = nn.Linear(input_size, hidden_size)
        self.embedding_tgt = nn.Linear(output_size, hidden_size)

        # Initialize weights using Xavier uniform initialization
        #nn.init.xavier_uniform_(self.embedding_src.weight)
        #nn.init.xavier_uniform_(self.embedding_tgt.weight)

        self.transformer = nn.Transformer(d_model=hidden_size, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, src, tgt):
        src = self.embedding_src(src)
        tgt = self.embedding_tgt(tgt)
        output = self.transformer(src, tgt)
        return self.fc_out(output)



def train_and_evaluate_model():
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Start timer
    start_time = time.time()

    # Load data
    # train_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/train_dataset.csv')
    # test_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/test_dataset.csv')
    train_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/train_dataset_small.csv')
    test_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/test_dataset_small.csv')


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
                             num_encoder_layers=NUM_ENCODER_LAYERS,
                             num_decoder_layers=NUM_DECODER_LAYERS).to(device)

    criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
            # Check gradients before the optimizer step
            print(f"Epoch {epoch}, Batch Gradients:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"    Gradient of {name}: {param.grad}")
                    if torch.isnan(param.grad).any():
                        print(f"    !!! NaN detected in gradients of {name}")


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            # Check weights after the optimizer step
            print(f"Epoch {epoch}, Batch Weights:")
            for name, param in model.named_parameters():
                print(f"    Weight of {name}: {param.data}")
                if torch.isnan(param.data).any():
                    print(f"    !!! NaN detected in weights of {name}")

            # Optionally, break if NaN is detected
            if torch.isnan(loss).any() or torch.isnan(outputs).any():
                print("!!! NaN detected, stopping training")
                break

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
    torch.save(model.state_dict(), 'transformer_weights.pth')
    torch.save(model, 'transformer.pth')
