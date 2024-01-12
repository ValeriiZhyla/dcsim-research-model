import numpy as np
import pandas as pd
import seaborn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


def pad_sequence(seq, window_size, padding_value=0):
    pad_length = window_size - len(seq)
    return np.pad(seq, ((0, pad_length), (0, 0)), mode='constant', constant_values=padding_value)


def create_windows_jobs(df, window_size, overlap_size, input_columns, output_columns):
    windowed_data = []

    # Group by simulation_id
    grouped = df.groupby('simulation_id', sort=False)
    print(f"Grouped by [simulation_id]. {grouped.ngroups} groups were found in dataset.")

    for name, group in grouped:
        sequence_length = len(group)
        if sequence_length < window_size:
            # Pad short sequences with zeros
            input_data = pad_sequence(group[input_columns].values, window_size)
            output_data = pad_sequence(group[output_columns].values, window_size)
            windowed_data.append((input_data, output_data))
        else:
            last_start = sequence_length - window_size
            for start in range(0, last_start + 1, window_size - overlap_size):
                end = start + window_size
                window = group.iloc[start:end]
                input_data = window[input_columns].values
                output_data = window[output_columns].values
                windowed_data.append((input_data, output_data))

            # Handle the last part of the sequence with padding if necessary
            if last_start + window_size < sequence_length:
                last_window = group.iloc[last_start:]
                input_data = pad_sequence(last_window[input_columns].values, window_size)
                output_data = pad_sequence(last_window[output_columns].values, window_size)
                windowed_data.append((input_data, output_data))
    return windowed_data

def pad_window_aux(df, window_size, columns):
    # Calculate the number of rows to pad
    pad_rows = window_size - len(df)

    # Check if padding is necessary
    if pad_rows > 0:
        # Create a padding array of zeros
        padding = np.zeros((pad_rows, len(columns)))
        # Append the padding to the sequence
        padded_sequence = np.vstack((df, padding))
    else:
        # No padding needed, just truncate the sequence
        padded_sequence = df[:window_size]

    return padded_sequence


def scale_and_reshape_jobs_windows(windowed_data, window_size, input_scaler, output_scaler, input_columns, output_columns):
    inputs, outputs = zip(*windowed_data)

    # Flatten, scale, and un-flatten inputs
    inputs_flattened = np.concatenate(inputs).reshape(-1, len(input_columns))
    inputs_scaled = input_scaler.transform(inputs_flattened)
    inputs_3d = inputs_scaled.reshape(-1, window_size, len(input_columns))

    # Flatten, scale, and un-flatten outputs
    outputs_flattened = np.concatenate(outputs).reshape(-1, len(output_columns))
    outputs_scaled = output_scaler.transform(outputs_flattened)
    outputs_3d = outputs_scaled.reshape(-1, window_size, len(output_columns))

    # Convert to tensors
    inputs_tensor = torch.tensor(inputs_3d, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs_3d, dtype=torch.float32)

    return TensorDataset(inputs_tensor, outputs_tensor)


def scale_and_reshape_aux_windows(windowed_data, window_size, scaler, aux_columns):
    # This function assumes windowed_data is already segmented into windows
    aux_data = np.array(windowed_data)

    # Flatten, scale, and un-flatten aux data
    aux_flattened = aux_data.reshape(-1, len(aux_columns))
    aux_scaled = scaler.transform(aux_flattened)
    aux_3d = aux_scaled.reshape(-1, window_size, len(aux_columns))

    # Convert to tensors
    aux_tensor = torch.tensor(aux_3d, dtype=torch.float32)
    return aux_tensor


def create_and_fit_scaler(df, columns):
    scaler = StandardScaler()

    features = df[columns].values

    scaler.fit(features)

    return scaler
