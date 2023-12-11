import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Constants
NUM_EPOCHS = 1000


# Define the LSTM Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        return out


# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load data
train_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/train_dataset.csv')
test_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/test_dataset.csv')

# Normalize data
input_scaler = StandardScaler()
output_scaler = StandardScaler()

# Fit the scalers on the training data
train_input_features = train_df[['flops', 'input_files_size', 'output_files_size']]
train_output_features = train_df[['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']]

input_scaler.fit(train_input_features)
output_scaler.fit(train_output_features)

# Transform training data
train_inputs = input_scaler.transform(train_input_features)
train_outputs = output_scaler.transform(train_output_features)

# Transform test data (using the same scalers)
test_input_features = test_df[['flops', 'input_files_size', 'output_files_size']]
test_output_features = test_df[['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']]

test_inputs = input_scaler.transform(test_input_features)
test_outputs = output_scaler.transform(test_output_features)

# Convert to PyTorch tensors
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_outputs = torch.tensor(train_outputs, dtype=torch.float32)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_outputs = torch.tensor(test_outputs, dtype=torch.float32)

train_inputs = train_inputs.unsqueeze(1)

# Create DataLoader instances
train_dataset = TensorDataset(train_inputs, train_outputs)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Initialize the model, loss function, and optimizer
model = RNNModel(input_size=3, hidden_size=50, output_size=5).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified version)
for epoch in range(NUM_EPOCHS):
    for inputs, targets in train_loader:
        # Move data to the device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item()}')

# Evaluate the model with test data
# ... (similar to training loop but without backpropagation)

# Plotting convergence
# ... (record and plot the training loss over epochs)