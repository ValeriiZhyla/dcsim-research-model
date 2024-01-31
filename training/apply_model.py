import numpy as np
import torch

from training import metrics
from training import plotting
from training import commons


def find_indexes_of_all_duplicates_except_first_occurrence(array):
    # Step 1: Flatten the array rows to tuples
    tuples = [tuple(row) for row in array]

    # Step 2: Identify unique rows and their first occurrences
    unique_tuples, indices, inverse = np.unique(tuples, axis=0, return_index=True, return_inverse=True)

    # Step 3: Identify duplicate indices
    duplicate_indices = np.setdiff1d(np.arange(len(tuples)), indices)

    return duplicate_indices


def remove_duplicates(indexes_of_duplicates, predictions_array, actual_values_array, actual_inputs_array):
    predictions_array_cleaned = np.delete(predictions_array, indexes_of_duplicates, axis=0)
    actual_values_array_cleaned = np.delete(actual_values_array, indexes_of_duplicates, axis=0)
    actual_inputs_array_cleaned = np.delete(actual_inputs_array, indexes_of_duplicates, axis=0)

    return predictions_array_cleaned, actual_values_array_cleaned, actual_inputs_array_cleaned


def apply_model_to_data(model, model_name, results_directory, dataset_path, plot_color, purpose, input_columns, output_columns, batch_size, window_size, window_overlap_size,
                        create_plots=True, create_kde=True):
    test_loader, test_scalers = commons.load_test_data(dataset_path, input_columns, output_columns, batch_size, window_size, window_overlap_size)

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model.to(device)

    # Evaluate the model with test data
    predictions_array, actual_values_array, actual_inputs_array = commons.evaluate_model_get_predictions_and_actual_values(model, test_loader, device)

    # Find and remove all duplicates, that are caused by overlapping windows and padding
    indexes_of_duplicates = find_indexes_of_all_duplicates_except_first_occurrence(actual_inputs_array)
    predictions_array_cleaned, actual_values_array_cleaned, actual_inputs_array_cleaned = remove_duplicates(indexes_of_duplicates, predictions_array, actual_values_array,
                                                                                                            actual_inputs_array)
    # Calculate metrics for each output parameter and show them
    metrics.calculate_and_show_metrics(results_directory, model_name, purpose, output_columns, predictions_array_cleaned, actual_values_array_cleaned)

    if create_plots:
        # Denormalize and plot results for each parameter
        plotting.denorm_and_plot_predicted_actual(output_columns, test_scalers, predictions_array_cleaned, actual_values_array_cleaned, model_name, results_directory,
                                                  color_name=plot_color, purpose=purpose, add_help_line=True)

    if create_kde:
        # Create KDE plot
        plotting.plot_kde(output_columns, predictions_array_cleaned, actual_values_array_cleaned, model_name, results_directory, color_name=plot_color, purpose=purpose)
