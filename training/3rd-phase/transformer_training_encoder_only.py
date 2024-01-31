import time

import seaborn
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import metrics
import plotting
from training import commons

# Constants
NUM_EPOCHS = 100
WINDOW_SIZE = 200
WINDOW_OVERLAP_SIZE = 100
BATCH_SIZE = 64
HIDDEN_LAYERS = 128
INPUT_SIZE = 4
OUTPUT_SIZE = 5
NHEADS = 2  # Ensure this is a divisor of HIDDEN_LAYERS
NUM_ENCODER_LAYERS = 2

model_name = "Transformer"
plot_color = seaborn.color_palette("deep")[4]  # deep purple



class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead, num_encoder_layers):
        super(TransformerModel, self).__init__()
        # Separate linear layers to project source and target features to hidden dimension
        self.src_input_projection = nn.Linear(input_size, hidden_size)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Linear layer to project from hidden dimension to output size
        self.output_projection = nn.Linear(hidden_size, output_size)



    def forward(self, src):
        # Project input to hidden size
        src = self.src_input_projection(src)

        # Transformer processing
        encoded_features = self.transformer_encoder(src)

        # Directly project each element in the sequence to the output size
        output = self.output_projection(encoded_features)

        return output

TRAIN_PATH = '../../dataset_preparation/3rd-phase/train_dataset.csv'
TEST_PATH = '../../dataset_preparation/3rd-phase/test_dataset.csv'
# TRAIN_PATH = '../../dataset_preparation/3rd-phase/train_dataset_small.csv'
# TEST_PATH = '../../dataset_preparation/3rd-phase/test_dataset_small.csv'

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

    # Initialize the model
    model = TransformerModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_LAYERS,
                             output_size=OUTPUT_SIZE, nhead=NHEADS,
                             num_encoder_layers=NUM_ENCODER_LAYERS).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # most accurate results so far for 0.001

    # Initialize the scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training Loop
    model.train()  # Set the model to training mode
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward pass
            loss.backward()

            # It helps, nan otherwise
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
    torch.save(model.state_dict(), 'generated-models/transformer_weights.pth')
    torch.save(model, 'generated-models/transformer.pth')
