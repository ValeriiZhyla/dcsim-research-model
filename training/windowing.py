import random

import numpy as np


def create_windows(df, window_size, overlap_size, input_columns, output_columns, group_by_column='simulation_id'):
    windowed_data = []

    # Group by simulation_id
    grouped = df.groupby(group_by_column, sort=False)
    print(f"Grouped by [{group_by_column}]. {grouped.ngroups} groups were found in dataset.")


    for group_name, group in grouped:
        sequence_length = len(group)

        window_start_index_delta = window_size - overlap_size
        next_window_start = 0
        while next_window_start < sequence_length:
            start = next_window_start
            end = start + window_size

            window = group.iloc[start:end]
            # Pad window with zeros if last sequence part is too short
            input_data = pad_sequence(window[input_columns].values, window_size)
            output_data = pad_sequence(window[output_columns].values, window_size)
            windowed_data.append((input_data, output_data))

            next_window_start = start + window_start_index_delta

    return windowed_data


def pad_sequence(seq, window_size, padding_value=0):
    pad_length = window_size - len(seq)
    return np.pad(seq, ((0, pad_length), (0, 0)), mode='constant', constant_values=padding_value)
