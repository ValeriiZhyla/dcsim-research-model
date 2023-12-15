import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

# Constants
NUM_EPOCHS = 250


# Define the LSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)
        out_forward = out[:, -1, :self.hidden_size]
        out_backward = out[:, 0, self.hidden_size:]
        out_combined = torch.cat((out_forward, out_backward), dim=1)
        out = self.fc(out_combined)
        return out


def create_windows(df, window_size, overlap_size, input_columns, output_columns):
    windowed_data = []
    for start in range(0, len(df) - window_size + 1, window_size - overlap_size):
        window = df[start:start + window_size]
        input_data = window[input_columns].values
        output_data = window[output_columns].iloc[-1].values  # Assuming the output is at the last timestep
        windowed_data.append((input_data, output_data))
    return windowed_data


def process_windows(windowed_data, input_scaler, output_scaler):
    inputs, outputs = zip(*windowed_data)
    # Normalize data
    inputs_scaled = input_scaler.transform(np.array(inputs).reshape(-1, len(input_columns))).reshape(-1, 500, len(input_columns))
    outputs_scaled = output_scaler.transform(np.array(outputs))

    # Convert to tensors
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs_scaled, dtype=torch.float32)

    return TensorDataset(inputs_tensor, outputs_tensor)


# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load data
# train_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/train_dataset.csv')
# test_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/test_dataset.csv')
train_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/train_dataset_small.csv')
test_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/test_dataset_small.csv')

# Transform dataframes into overlapping windows
input_columns = ['flops', 'input_files_size', 'output_files_size']
output_columns = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']
train_windows = create_windows(train_df, window_size=500, overlap_size=250, input_columns=input_columns, output_columns=output_columns)
test_windows = create_windows(test_df, window_size=500, overlap_size=250, input_columns=input_columns, output_columns=output_columns)

# Fit the scalers on the whole training dataset
input_scaler = StandardScaler()
output_scaler = StandardScaler()

train_input_features = train_df[input_columns].values
train_output_features = train_df[output_columns].values

input_scaler.fit(train_input_features)
output_scaler.fit(train_output_features)

# Prepare datasets
train_dataset = process_windows(train_windows, input_scaler, output_scaler)
test_dataset = process_windows(test_windows, input_scaler, output_scaler)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = BiLSTMModel(input_size=3, hidden_size=50, output_size=5).to(device)
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

print("hello")
print("hello")
# Plotting convergence
# ... (record and plot the training loss over epochs)
