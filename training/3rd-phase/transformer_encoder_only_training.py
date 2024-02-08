import time

import seaborn
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training import commons

# Constants
NUM_EPOCHS = 250
WINDOW_SIZE = 200
WINDOW_OVERLAP_SIZE = 100
BATCH_SIZE = 64
HIDDEN_SIZE = 8

NHEADS = 2  # Ensure this is a divisor of hidden_size
NUM_ENCODER_LAYERS = 2

INPUT_SIZE = 6
OUTPUT_SIZE = 5

model_name = "Transformer"
plot_color = seaborn.color_palette("deep")[4]  # deep purple



class TransformerEncoderOnly(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead, num_encoder_layers):
        super(TransformerEncoderOnly, self).__init__()
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

input_columns = ['simulation_id_int', 'simulation_length', 'index', 'flops', 'input_files_size', 'output_files_size']
output_columns = ['job_start', 'job_end', 'compute_time', 'input_files_transfer_time', 'output_files_transfer_time']


def train_and_evaluate_model(num_epochs, window_size, window_overlap, batch_size, hidden_size, nheads, encoder_layers):
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Start timer
    start_time = time.time()

    # Load data and scalers
    train_loader, train_scalers, test_loader, test_scalers = commons.load_data(TRAIN_PATH, TEST_PATH, input_columns, output_columns, batch_size, window_size, window_overlap)

    # Initialize the model
    model = TransformerEncoderOnly(input_size=INPUT_SIZE, hidden_size=hidden_size,
                                   output_size=OUTPUT_SIZE, nhead=nheads,
                                   num_encoder_layers=encoder_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # most accurate results so far for 0.001

    # Initialize the scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training Loop
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss}')
        scheduler.step(avg_loss)

    # Stop timer and print training summary
    end_time = time.time()
    total_time = end_time - start_time
    commons.print_training_summary(num_epochs, window_size, window_overlap, batch_size, hidden_size, total_time)

    return model


if __name__ == '__main__':
    model = train_and_evaluate_model(NUM_EPOCHS, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, HIDDEN_SIZE, NHEADS, NUM_ENCODER_LAYERS)
    torch.save(model.state_dict(), 'generated-models/default/transformer_encoder_only_weights.pth')
    torch.save(model, 'generated-models/default/transformer_encoder_only.pth')
