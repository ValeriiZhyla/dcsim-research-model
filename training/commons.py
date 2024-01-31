import os

import numpy as np
import pandas as pd
import torch
import json
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from training import windowing

HYPERPARAMETERS_FILE = 'hyperparameters.json'
GENERATED_MODELS_DIRECTORY = 'generated-models'


def load_data(path_train, path_test, input_columns, output_columns, batch_size, window_size, window_overlap_size):
    train_df = pd.read_csv(path_train, delimiter=';')
    test_df = pd.read_csv(path_test, delimiter=';')

    # Fit the scalers on the whole training dataset
    train_scalers, train_df_scaled = df_fit_transform_and_get_scalers(train_df, input_columns + output_columns)
    test_scalers, test_df_scaled = df_fit_transform_and_get_scalers(test_df, input_columns + output_columns)

    train_windows = windowing.create_windows(train_df_scaled, window_size=window_size, overlap_size=window_overlap_size, input_columns=input_columns, output_columns=output_columns)
    test_windows = windowing.create_windows(test_df_scaled, window_size=window_size, overlap_size=window_overlap_size, input_columns=input_columns, output_columns=output_columns)

    train_dataset = create_tensor_dataset(train_windows)
    test_dataset = create_tensor_dataset(test_windows)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, train_scalers, test_loader, test_scalers


def load_test_data(path_test, input_columns, output_columns, batch_size, window_size, window_overlap_size):
    test_df = pd.read_csv(path_test, delimiter=';')

    # Fit the scalers on the whole training dataset
    test_scalers, test_df_scaled = df_fit_transform_and_get_scalers(test_df, input_columns + output_columns)

    test_windows = windowing.create_windows(test_df_scaled, window_size=window_size, overlap_size=window_overlap_size, input_columns=input_columns, output_columns=output_columns)

    test_dataset = create_tensor_dataset(test_windows)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, test_scalers


def create_tensor_dataset(windowed_data):
    # Convert windowed data to tensors
    inputs, outputs = zip(*windowed_data)

    # Converting lists of numpy arrays to numpy arrays
    inputs_array = np.array(inputs, dtype=np.float32)
    outputs_array = np.array(outputs, dtype=np.float32)

    # Converting numpy arrays to tensors
    inputs_tensor = torch.tensor(inputs_array, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs_array, dtype=torch.float32)

    # Create TensorDataset
    dataset = TensorDataset(inputs_tensor, outputs_tensor)
    return dataset


def df_fit_transform_and_get_scalers(df, columns_to_scale):
    scalers = {col: StandardScaler() for col in columns_to_scale}

    for col, scaler in scalers.items():
        df.loc[:, col] = scaler.fit_transform(df[[col]])

    return scalers, df


def evaluate_model_get_predictions_and_actual_values(model, test_loader, device):
    # Evaluate the model with test data
    model.eval()
    predictions = []

    actual_inputs = []
    actual_outputs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to the device
            inputs = inputs.float().to(device)
            targets = targets.float()

            # Make a prediction
            outputs = model(inputs)

            # Store predictions and actual values for further metrics calculations
            predictions.extend(outputs.cpu().numpy())
            actual_outputs.extend(targets.cpu().numpy())
            # We will need inputs to reverse the windowing process
            actual_inputs.extend(inputs.cpu().numpy())

    # Convert lists of arrays to single numpy arrays
    predictions_array = np.vstack(predictions)
    actual_values_array = np.vstack(actual_outputs)
    actual_inputs_array = np.vstack(actual_inputs)

    return predictions_array, actual_values_array, actual_inputs_array


def print_training_summary(num_epochs, window_size, window_overlap_size, batch_size, hidden_layers, total_time):
    print("=================================")
    print(f"Epochs: {num_epochs}")
    print(f"Window size: {window_size}")
    print(f"Window overlap: {window_overlap_size}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Total time for training: {total_time:.2f} seconds")
    print("=================================")


def generate_directory_name_with_hyperparameters(model_name, epochs, window_size, window_overlap, batch_size, hidden_layers, heads=0, encoders=0, decoders=0) -> str:
    if "transformer" in model_name:
        return f"{model_name}_{epochs}ep_{window_size}wd_{window_overlap}wo_{batch_size}bat_{hidden_layers}hl_{heads}hd_{encoders}enc_{decoders}dec"
    else:
        return f"{model_name}_{epochs}ep_{window_size}wd_{window_overlap}wo_{batch_size}bat_{hidden_layers}hl"


def directory_name_with_hyperparameters_already_exists(model_name, epochs, window_size, window_overlap, batch_size, hidden_layers, heads=0, encoders=0, decoders=0) -> bool:
    directory = generate_directory_name_with_hyperparameters(model_name, epochs, window_size, window_overlap, batch_size, hidden_layers, heads, encoders, decoders)
    full_path = os.path.join(GENERATED_MODELS_DIRECTORY, directory)

    if os.path.exists(full_path):
        print(f"{full_path} already exists")
        return True
    else:
        return False


def save_model_hyperparameters(directory, window_size, window_overlap, batch_size, hidden_layers, heads=0, encoders=0, decoders=0):
    hyperparameters = {'window_size': window_size, 'window_overlap': window_overlap, 'batch_size': batch_size, 'hidden_layers': hidden_layers, 'heads': heads, 'encoders': encoders,
                       'decoders': decoders}
    with open(os.path.join(directory, HYPERPARAMETERS_FILE), "w") as file:
        json.dump(hyperparameters, file)


def save_model_and_get_directory(model, model_name, epochs, window_size, window_overlap, batch_size, hidden_layers, heads=0, encoders=0, decoders=0) -> str:
    directory = generate_directory_name_with_hyperparameters(model_name, epochs, window_size, window_overlap, batch_size, hidden_layers, heads, encoders, decoders)
    full_path = os.path.join(GENERATED_MODELS_DIRECTORY, directory)

    # create the directory if it doesn't exist
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    # save the model in this directory
    torch.save(model.state_dict(), os.path.join(full_path, f'{model_name}_weights.pth'))
    torch.save(model, os.path.join(full_path, f'{model_name}.pth'))

    # save the model hyperparameters
    save_model_hyperparameters(full_path, window_size, window_overlap, batch_size, hidden_layers, heads, encoders, decoders)

    return full_path
