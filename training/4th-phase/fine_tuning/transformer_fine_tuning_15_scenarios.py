import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from initial_training.transformer_initial_training import TransformerModelWithTwoAuxEncoders, CombinedDataset, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE

import windowing






# Function to check for NaN in gradients
def check_nan_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")

def check_nan_in_data(*args):
    for i, data in enumerate(args):
        if torch.isnan(data).any():
            print(f"NaN detected in input data at position {i}")


def fine_tune_model(model: TransformerModelWithTwoAuxEncoders):
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Start timer
    start_time = time.time()

    # scenarios.reverse() # Try desy first: 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3 OK
    # Simple jobs are killing! -> very small job start, near 0. Add +10 -> not helping

    for _ in range(0, NUM_ITERATIONS_OVER_ALL_SCENARIOS):
        for scenario in scenarios:
            # Load dataframes
            train_df, test_df, nodes_aux_df, links_aux_df = scenario.read_all_datasets()

            # Load jobs
            train_windows = windowing.create_windows_jobs(train_df, window_size=WINDOW_SIZE, overlap_size=WINDOW_OVERLAP_SIZE, input_columns=input_columns_jobs,
                                                          output_columns=output_columns_jobs)

            # Load aux data
            nodes_aux_df = nodes_aux_df[nodes_columns]
            links_aux_df = links_aux_df[links_columns]
            nodes_aux_windows = windowing.pad_window_aux(nodes_aux_df, window_size=WINDOW_SIZE, columns=nodes_columns)
            links_aux_windows = windowing.pad_window_aux(links_aux_df, window_size=WINDOW_SIZE, columns=links_columns)

            # Fit the scalers on the whole training dataset
            input_jobs_scaler = windowing.create_and_fit_scaler(train_df, input_columns_jobs)
            output_jobs_scaler = windowing.create_and_fit_scaler(train_df, output_columns_jobs)
            nodes_scaler = windowing.create_and_fit_scaler(nodes_aux_df, nodes_columns)
            links_scaler = windowing.create_and_fit_scaler(links_aux_df, links_columns)

            # Prepare datasets
            train_dataset = windowing.scale_and_reshape_jobs_windows(train_windows, WINDOW_SIZE, input_jobs_scaler, output_jobs_scaler, input_columns_jobs, output_columns_jobs)
            nodes_aux_dataset = windowing.scale_and_reshape_aux_windows(nodes_aux_windows, WINDOW_SIZE, nodes_scaler, nodes_columns)
            links_aux_dataset = windowing.scale_and_reshape_aux_windows(links_aux_windows, WINDOW_SIZE, links_scaler, links_columns)

            train_dataset_jobs_with_aux = CombinedDataset(train_dataset, nodes_aux_dataset, links_aux_dataset)

            train_loader = DataLoader(train_dataset_jobs_with_aux, batch_size=BATCH_SIZE, shuffle=True)

            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)  # smaller learning rate

            # Training Loop
            model.train()  # Set the model to training mode
            last_avg_loss = 0
            for epoch in range(NUM_EPOCHS_ON_EACH_SCENARIO):
                total_loss = 0
                for batch in train_loader:
                    # Unpack the batch data
                    train_data, nodes_aux_data, links_aux_data, train_target = batch

                    # Move data to the device
                    train_data = train_data.to(device)
                    nodes_aux_data = nodes_aux_data.to(device)
                    links_aux_data = links_aux_data.to(device)
                    train_target = train_target.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    check_nan_in_data(train_data, nodes_aux_data, links_aux_data, train_target)

                    # Forward pass
                    outputs = model(train_data, nodes_aux_data, links_aux_data, train_target)

                    # Check for NaN in model outputs
                    if torch.isnan(outputs).any():
                        print("NaN detected in model outputs")

                    loss = criterion(outputs, train_target)

                    # Backward pass
                    loss.backward()
                    check_nan_gradients(model)  # Check for NaN in gradients
                    # It helps, nan otherwise
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                    optimizer.step()

                    # Accumulate loss
                    total_loss += loss.item()
                avg_loss = total_loss / len(train_loader)
                last_avg_loss = avg_loss
                print(f'{scenario.name}: Epoch [{epoch + 1}/{NUM_EPOCHS_ON_EACH_SCENARIO}], Average Loss: {avg_loss}')

            # Print training summary
            print("=================================")
            print(f"Scenario: {scenario.name}")
            print(f"Last loss: {last_avg_loss}")
            print("=================================")

    end_time = time.time()
    # Stop timer
    total_time = end_time - start_time
    print(f"Total time for training: {total_time:.2f} seconds")

    return model


if __name__ == '__main__':
    model = torch.load('../../../trained-models/4th-phase/initial_model/transformer.pth')

    model = fine_tune_model(model)
    torch.save(model.state_dict(), '../generated-models/transformer_tuned_on_15_scenarios-default_weights.pth')
    torch.save(model, '../generated-models/transformer_tuned_on_15_scenarios-default.pth')
