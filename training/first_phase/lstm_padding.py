import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

# Constants
NUM_EPOCHS = 20


# Define the LSTM Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)  # Use bidirectional LSTM, to capture the data in both directions
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiplied by 2 for bidirectional

    def forward(self, x, lengths):
        # Pack the input sequences
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Pass the packed sequence through the RNN
        out_packed, _ = self.rnn(x_packed)

        # Unpack the sequence
        out, _ = pad_packed_sequence(out_packed, batch_first=True)

        # Concatenate the hidden state from the last time step of both directions
        out_forward = torch.stack([out[i, length - 1, :self.hidden_size] for i, length in enumerate(lengths)])
        out_backward = out[:, 0, self.hidden_size:]  # First time step, backward direction

        out_combined = torch.cat((out_forward, out_backward), dim=1)
        out = self.fc(out_combined)
        return out

def collate_fn(batch):
    """ A custom collate function for DataLoader to handle padding of sequences with variable lengths. """
    batch.sort(key=lambda x: len(x[0]), reverse=True)  # Sort the batch by sequence length in descending order
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])  # Store the lengths of sequences
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)  # Pad the sequences
    labels = torch.stack(labels)  # Stack labels
    return sequences_padded, labels, lengths

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load data
# train_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/train_dataset.csv')
# test_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/test_dataset.csv')
train_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/train_dataset_small.csv')
test_df = pd.read_csv('../../simulation-dataset-preparation/first_phase/test_dataset_small.csv')

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
test_inputs = test_inputs.unsqueeze(1)

# Create DataLoader instances
train_dataset = TensorDataset(train_inputs, train_outputs)
test_dataset = TensorDataset(test_inputs, test_outputs)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# Initialize the model, loss function, and optimizer
model = RNNModel(input_size=3, hidden_size=10, output_size=5).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(NUM_EPOCHS):
    for inputs, targets, lengths in train_loader:
        # Move data to the device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
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
