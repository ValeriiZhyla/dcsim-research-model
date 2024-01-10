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


def create_windows(df, window_size, overlap_size, input_columns, output_columns):
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


def scale_and_reshape_windows(windowed_data, window_size, input_scaler, output_scaler, input_columns, output_columns):
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


def create_and_fit_scalers(df, input_columns, output_columns):
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    input_features = df[input_columns].values
    output_features = df[output_columns].values

    input_scaler.fit(input_features)
    output_scaler.fit(output_features)

    return input_scaler, output_scaler


def calculate_metrics(actual_values, predicted_values, output_columns):
    # Calculate metrics for each output parameter
    mse_per_param = []
    mae_per_param = []
    rmse_per_param = []
    r_squared_per_param = []

    for i in range(len(output_columns)):
        # Extract predictions and actual values for the current parameter
        actual_param = actual_values[:, i]
        predicted_param = predicted_values[:, i]

        # Calculate metrics for this parameter
        mse = mean_squared_error(actual_param, predicted_param)
        mae = mean_absolute_error(actual_param, predicted_param)
        rmse = np.sqrt(mse)
        r_squared = r2_score(actual_param, predicted_param)

        # Store the metrics
        mse_per_param.append(mse)
        mae_per_param.append(mae)
        rmse_per_param.append(rmse)
        r_squared_per_param.append(r_squared)

    return mse_per_param, mae_per_param, rmse_per_param, r_squared_per_param


def show_metrics(output_columns, mse_per_param, mae_per_param, rmse_per_param, r_squared_per_param):
    for i in range(len(output_columns)):
        print(f"Parameter [{output_columns[i]}]:")
        print(f"  Mean Squared Error (MSE):         {str(round(mse_per_param[i], 7)).rjust(15)}")
        print(f"  Root Mean Squared Error (RMSE):   {str(round(rmse_per_param[i], 7)).rjust(15)}")
        print(f"  Mean Absolute Error (MAE):        {str(round(mae_per_param[i], 7)).rjust(15)}")
        print(f"  R-squared (R^2):                  {str(round(r_squared_per_param[i], 7)).rjust(15)}")
        print("=================================")


def denorm_and_plot_predicted_actual(output_columns, output_scaler, predictions_array, actual_values_array, model_name, max_points=10000000, color_name="blue", purpose=None, add_help_line=True):
    # Scale back to original values for readable plots
    denorm_predictions_array = output_scaler.inverse_transform(predictions_array)
    denorm_actual_values_array = output_scaler.inverse_transform(actual_values_array)

    # Check if the number of points is more than max_points
    if len(predictions_array) > max_points:
        # Randomly select max_points indices
        indices = np.random.choice(len(predictions_array), size=max_points, replace=False)
        denorm_predictions_array = denorm_predictions_array[indices]
        denorm_actual_values_array = denorm_actual_values_array[indices]

    seaborn.set_style("whitegrid")

    # Create a DataFrame for plotting
    for i, label in enumerate(output_columns):
        # Create a DataFrame for plotting
        data_for_plot = pd.DataFrame({
            'Actual Values': denorm_actual_values_array[:, i],
            'Predictions': denorm_predictions_array[:, i]
        })

        # Create the scatter plot
        plt.figure(figsize=(6, 6))
        plot = seaborn.scatterplot(data=data_for_plot, x='Actual Values', y='Predictions', alpha=0.1, color=color_name)
        plt.title(f'{label} [{model_name}]', fontweight='bold', pad=20, fontsize=14)
        plot.set_xlabel(plot.get_xlabel(), fontdict={'weight': 'bold'}, labelpad=15, fontsize=14),
        plot.set_ylabel(plot.get_ylabel(), fontdict={'weight': 'bold'}, labelpad=15, fontsize=14)

        if purpose is None:
            file_name = f"plots/{model_name}-{label}.png"
        else:
            file_name = f"plots/{model_name}-{purpose}-{label}.png"

        if add_help_line:
            # Get current axis
            ax = plt.gca()
            # Get limits for the current axis
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # Set the limits for x and y to be the same for the diagonal line
            lim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
            ax.plot(lim, lim, color='#808080', alpha=0.75, zorder=3, label='Ideal Prediction')  # Medium-grey color
            # Reset the limits to original
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            # Add legend to the plot
            plt.legend()

        plt.savefig(file_name, dpi=200, bbox_inches='tight', format='png')
        plt.show()


def denorm_and_plot_kde(output_columns, output_scaler, predictions_array, actual_values_array, model_name, color_name="blue", purpose=None):
    # Your existing code for scaling and selecting data...

    denorm_predictions_array = output_scaler.inverse_transform(predictions_array)
    denorm_actual_values_array = output_scaler.inverse_transform(actual_values_array)

    seaborn.set_style("whitegrid")

    for i, label in enumerate(output_columns):
        # Create a DataFrame for plotting
        data_for_plot = pd.DataFrame({
            'Actual Values': denorm_actual_values_array[:, i],
            'Predictions': denorm_predictions_array[:, i]
        })

        # Create figure for KDE plots
        plt.figure(figsize=(8, 6))

        # KDE plot for actual values
        seaborn.kdeplot(data=data_for_plot, x='Actual Values', label='Actual Values', color='green', fill=True)

        # KDE plot for predictions
        seaborn.kdeplot(data=data_for_plot, x='Predictions', label='Predictions', color=color_name, fill=True)

        # Set title and labels
        plt.title(f'{label} KDE [{model_name}]', fontweight='bold', fontsize=14)
        plt.xlabel('Values', fontweight='bold', fontsize=14)
        plt.ylabel('Density', fontweight='bold', fontsize=14)

        # Add legend
        plt.legend()

        # Determine file name based on purpose
        if purpose is None:
            file_name = f"plots/{model_name}-kde-{label}.png"
        else:
            file_name = f"plots/{model_name}-{purpose}-kde-{label}.png"

        # Save plot
        plt.savefig(file_name, dpi=200, bbox_inches='tight')

        # Show plot
        plt.show()

def calculate_and_show_metrics(output_columns, predictions_array, actual_values_array):
    mse_per_param, mae_per_param, rmse_per_param, r_squared_per_param = calculate_metrics(actual_values_array, predictions_array, output_columns)
    show_metrics(output_columns, mse_per_param, mae_per_param, rmse_per_param, r_squared_per_param)
