import time

import seaborn
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import metrics
import plotting
from training import commons

# Constants
NUM_EPOCHS = 200
WINDOW_SIZE = 200
WINDOW_OVERLAP_SIZE = 100
BATCH_SIZE = 128
HIDDEN_LAYERS = 125
INPUT_SIZE = 4
OUTPUT_SIZE = 5

model_name = "LSTM"
plot_color = seaborn.color_palette("deep")[1]  # deep orange


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


TRAIN_PATH = '../../dataset-preparation/2nd-phase/train_dataset.csv'
TEST_PATH = '../../dataset-preparation/2nd-phase/test_dataset.csv'
# TRAIN_PATH = '../../dataset-preparation/2nd-phase/train_dataset_small.csv'
# TEST_PATH = '../../dataset-preparation/2nd-phase/test_dataset_small.csv'

input_columns = ['index', 'flops', 'input_files_size', 'output_files_size']
output_columns = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']

def train_and_evaluate_model():
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Start timer
    start_time = time.time()

    # Load data and scalers
    train_loader, train_scalers, test_loader, test_scalers = commons.load_data(TRAIN_PATH, TEST_PATH, input_columns, output_columns, BATCH_SIZE, WINDOW_SIZE, WINDOW_OVERLAP_SIZE)

    # Initialize the model, loss function, and optimizer
    model = BiLSTMModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_LAYERS, output_size=OUTPUT_SIZE).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

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
        scheduler.step(avg_loss)

    # Stop timer and print training summary
    end_time = time.time()
    total_time = end_time - start_time
    commons.print_training_summary(NUM_EPOCHS, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, HIDDEN_LAYERS, total_time)

    # Evaluate the model with test data
    predictions_array, actual_values_array = commons.evaluate_model_get_predictions_and_actual_values(model, test_loader, device)

    # Calculate metrics for each output parameter and show them
    metrics.calculate_and_show_metrics(output_columns, predictions_array, actual_values_array)

    # Denormalize and plot results for each parameter
    plotting.denorm_and_plot_predicted_actual(output_columns, test_scalers, predictions_array, actual_values_array, model_name, purpose="training")
    return model


if __name__ == '__main__':
    model = train_and_evaluate_model()
    torch.save(model.state_dict(), 'generated-models/lstm_weights.pth')
    torch.save(model, 'generated-models/lstm.pth')