import os

import numpy as np
import pandas as pd
import seaborn
import torch
import json
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

HYPERPARAMETERS_FILE = 'hyperparameters.json'
GENERATED_MODELS_DIRECTORY = 'generated-models'


def load_data(path_train, path_test, input_columns, output_columns, batch_size, window_size, window_overlap_size):
    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)

    # Fit the scalers on the whole training dataset
    train_scalers, train_df_scaled = df_fit_transform_and_get_scalers(train_df, input_columns + output_columns)
    test_scalers, test_df_scaled = df_fit_transform_and_get_scalers(test_df, input_columns + output_columns)

    train_windows = create_windows(train_df_scaled, window_size=window_size, overlap_size=window_overlap_size, input_columns=input_columns, output_columns=output_columns)
    test_windows = create_windows(test_df_scaled, window_size=window_size, overlap_size=window_overlap_size, input_columns=input_columns, output_columns=output_columns)

    train_dataset = create_tensor_dataset(train_windows)
    test_dataset = create_tensor_dataset(test_windows)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, train_scalers, test_loader, test_scalers


def load_test_data(path_test, input_columns, output_columns, batch_size, window_size, window_overlap_size):
    test_df = pd.read_csv(path_test)

    # Fit the scalers on the whole training dataset
    test_scalers, test_df_scaled = df_fit_transform_and_get_scalers(test_df, input_columns + output_columns)

    test_windows = create_windows(test_df_scaled, window_size=window_size, overlap_size=window_overlap_size, input_columns=input_columns, output_columns=output_columns)

    test_dataset = create_tensor_dataset(test_windows)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, test_scalers


def create_tensor_dataset(windowed_data):
    # Convert windowed data to tensors
    inputs, outputs = zip(*windowed_data)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

    # Create TensorDataset
    dataset = TensorDataset(inputs_tensor, outputs_tensor)
    return dataset


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


def df_fit_transform_and_get_scalers(df, columns_to_scale):
    scalers = {col: StandardScaler() for col in columns_to_scale}

    for col, scaler in scalers.items():
        df.loc[:, col] = scaler.fit_transform(df[[col]])

    return scalers, df


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


def show_and_save_metrics(results_directory, purpose, output_columns, mse_per_param, mae_per_param, rmse_per_param, r_squared_per_param):
    lines = []
    for i in range(len(output_columns)):
        lines.append(f"Parameter [{output_columns[i]}]:")
        lines.append(f"  Mean Squared Error (MSE):         {str(round(mse_per_param[i], 7)).rjust(15)}")
        lines.append(f"  Root Mean Squared Error (RMSE):   {str(round(rmse_per_param[i], 7)).rjust(15)}")
        lines.append(f"  Mean Absolute Error (MAE):        {str(round(mae_per_param[i], 7)).rjust(15)}")
        lines.append(f"  R-squared (R^2):                  {str(round(r_squared_per_param[i], 7)).rjust(15)}")
        lines.append("=================================")

    for line in lines:
        print(line)

    file_name = os.path.join(results_directory, f"metrics_{purpose}.txt")
    with open(file_name, 'w') as file:
        for line in lines:
            file.write(line + '\n')
    print("Metrics saved to {}".format(file_name))



def save_metrics_latex(results_directory, model_name, purpose, output_columns, mse_per_param, mae_per_param, rmse_per_param, r_squared_per_param):
    p = 5
    lines_start = [r"\begin{table}[ht!]",
             r"\centering",
             r"\caption{" + model_name + ": " + purpose.capitalize() + " accuracy metrics}",
             r"\label{table:TODO-PHASE-LABEL/" + model_name.lower() + "-" + purpose.lower() + "-metrics}",
             r"\renewcommand{\arraystretch}{1.25}",
             r"\begin{tabular}{|c||c|>{\columncolor{lightgray}}c|c|>{\columncolor{lightgray}}c|}",
             r"\hline",
             r"%\rowcolor{white} % Keeps the header row white",
             r"\textbf{} & \textbf{MSE} & \textbf{RMSE} & \textbf{MAE} & \textbf{R\textsuperscript{2}} \\",
             r"\hline",
             r"\hline"]

    lines_end = [
        r"\end{tabular}",
        r"\end{table}"
    ]

    lines_body = []
    for i in range(len(output_columns)):
        name = output_columns[i]
        name = name.replace("_", "\_")
        line = f"{name} & {str(round(mse_per_param[i], p))} & {str(round(rmse_per_param[i], p))} & {str(round(mae_per_param[i], p))} & {str(round(r_squared_per_param[i], p))} \\\\"
        lines_body.append(line)
        lines_body.append(r"\hline")

    lines = lines_start + lines_body + lines_end

    file_name = os.path.join(results_directory, f"metrics_{purpose}_latex.txt")
    with open(file_name, 'w') as file:
        for line in lines:
            file.write(line + '\n')
    print("Latex table saved to {}".format(file_name))


def denorm_and_plot_predicted_actual(output_columns, output_scalers, predictions_array, actual_values_array, model_name, results_directory, max_points=1000000, color_name="blue",
                                     purpose=None,
                                     add_help_line=True):
    # Initialize arrays for denormalized data
    denorm_predictions_array = np.zeros_like(predictions_array)
    denorm_actual_values_array = np.zeros_like(actual_values_array)

    # Denormalize each column separately
    for i, col in enumerate(output_columns):
        scaler = output_scalers[col]
        denorm_predictions_array[:, i] = scaler.inverse_transform(predictions_array[:, i].reshape(-1, 1)).flatten()
        denorm_actual_values_array[:, i] = scaler.inverse_transform(actual_values_array[:, i].reshape(-1, 1)).flatten()

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

        plots_directory = os.path.join(results_directory, os.path.join("plots", os.path.join("accuracy", purpose)))
        if not os.path.exists(plots_directory):
            os.makedirs(plots_directory)

        if purpose is None:
            file_name = os.path.join(plots_directory, f"{model_name}-{label}.png")
        else:
            file_name = os.path.join(plots_directory, f"{model_name}-{purpose}-{label}.png")

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


def plot_kde(output_columns, predictions_array, actual_values_array, model_name, results_directory, color_name="blue", purpose=None):
    seaborn.set_style("whitegrid")

    for i, label in enumerate(output_columns):
        # Create a DataFrame for plotting
        data_for_plot = pd.DataFrame({
            'Actual Values': actual_values_array[:, i],
            'Predictions': predictions_array[:, i]
        })

        # Create figure for KDE plots
        plt.figure(figsize=(8, 6))

        # KDE plot for actual values
        seaborn.kdeplot(data=data_for_plot, x='Actual Values', label='Actual Values', color='green', fill=True, linewidth=2)

        # KDE plot for predictions
        seaborn.kdeplot(data=data_for_plot, x='Predictions', label='Predictions', color=color_name, fill=True, linewidth=2)

        # Set title and labels
        plt.title(f'{label} KDE [{model_name}]', fontweight='bold', fontsize=14)
        plt.xlabel('Values', fontweight='bold', fontsize=14)
        plt.ylabel('Density', fontweight='bold', fontsize=14)

        # Add legend
        plt.legend()

        plots_directory = os.path.join(results_directory, os.path.join("plots", os.path.join("kde", purpose)))
        if not os.path.exists(plots_directory):
            os.makedirs(plots_directory)

        # Determine file name based on purpose
        if purpose is None:
            file_name = os.path.join(plots_directory, f"{model_name}-kde-{label}.png")
        else:
            file_name = os.path.join(plots_directory, f"{model_name}-{purpose}-kde-{label}.png")

        # Save plot
        plt.savefig(file_name, dpi=200, bbox_inches='tight')

        # Show plot
        plt.show()


def calculate_and_show_metrics(results_directory, model_name, purpose, output_columns, predictions_array, actual_values_array):
    mse_per_param, mae_per_param, rmse_per_param, r_squared_per_param = calculate_metrics(actual_values_array, predictions_array, output_columns)
    show_and_save_metrics(results_directory, purpose, output_columns, mse_per_param, mae_per_param, rmse_per_param, r_squared_per_param)
    save_metrics_latex(results_directory, model_name, purpose, output_columns, mse_per_param, mae_per_param, rmse_per_param, r_squared_per_param)


def evaluate_model_get_predictions_and_actuals(model, test_loader, device):
    # Evaluate the model with test data
    model.eval()
    predictions = []
    actual_values = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to the device
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)

            # Make a prediction
            outputs = model(inputs)

            # Store predictions and actual values for further metrics calculations
            predictions.extend(outputs.cpu().numpy())
            actual_values.extend(targets.cpu().numpy())

    # Convert lists of arrays to single numpy arrays
    predictions_array = np.vstack(predictions)
    actual_values_array = np.vstack(actual_values)

    return predictions_array, actual_values_array


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
