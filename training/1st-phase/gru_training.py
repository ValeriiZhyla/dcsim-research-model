import time

import seaborn
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from training import commons

# Constants
NUM_EPOCHS = 200
WINDOW_SIZE = 200
WINDOW_OVERLAP_SIZE = 100
BATCH_SIZE = 32
HIDDEN_SIZE = 100
INPUT_SIZE = 5
OUTPUT_SIZE = 5
LAYERS = 1

model_name = "GRU"
plot_color = seaborn.color_palette("deep")[0]  # deep blue


# Define the GRU Model
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    def forward(self, x):
        # GRU layer
        out, _ = self.gru(x)  # out shape: [batch_size, sequence_length, hidden_size * 2]

        # Apply the linear layer to each time step
        out = self.fc(out)  # out shape: [batch_size, sequence_length, output_size]
        return out


TRAIN_PATH = '../../dataset_preparation/1st-phase/train_dataset.csv'
TEST_PATH = '../../dataset_preparation/1st-phase/test_dataset.csv'
# TRAIN_PATH = '../../dataset_preparation/1st-phase/train_dataset_small.csv'
# TEST_PATH = '../../dataset_preparation/1st-phase/test_dataset_small.csv'

input_columns = ['simulation_length', 'index', 'flops', 'input_files_size', 'output_files_size']
output_columns = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']


def train_and_evaluate_model(num_epochs, window_size, window_overlap, batch_size, hidden_size, layers):
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Start timer
    start_time = time.time()

    # Load data and scalers
    train_loader, train_scalers, test_loader, test_scalers = commons.load_data(TRAIN_PATH, TEST_PATH, input_columns, output_columns, batch_size, window_size, window_overlap)

    # Initialize the model, loss function, and optimizer
    model = BiGRUModel(input_size=INPUT_SIZE, hidden_size=hidden_size, output_size=OUTPUT_SIZE, num_layers=layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss}')
        scheduler.step(avg_loss)

    # Stop timer and print training summary
    end_time = time.time()
    total_time = end_time - start_time
    commons.print_training_summary(num_epochs, window_size, window_overlap, batch_size, hidden_size, total_time)

    return model


if __name__ == '__main__':
    model = train_and_evaluate_model(NUM_EPOCHS, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, HIDDEN_SIZE, LAYERS)
    torch.save(model.state_dict(), 'generated-models/default/gru_weights.pth')
    torch.save(model, 'generated-models/default/gru.pth')