import torch

from training import metrics
from training import plotting
from training import commons

def apply_model_to_data(model, model_name, results_directory, dataset_path, plot_color, purpose, input_columns, output_columns, batch_size, window_size, window_overlap_size, create_plots=True, create_kde=True):
    test_loader, test_scalers = commons.load_test_data(dataset_path, input_columns, output_columns, batch_size, window_size, window_overlap_size)

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model.to(device)

    # Evaluate the model with test data
    predictions_array, actual_values_array = commons.evaluate_model_get_predictions_and_actual_values(model, test_loader, device)

    # Calculate metrics for each output parameter and show them
    metrics.calculate_and_show_metrics(results_directory, model_name, purpose, output_columns, predictions_array, actual_values_array)


    if create_plots:
        # Denormalize and plot results for each parameter
        plotting.denorm_and_plot_predicted_actual(output_columns, test_scalers, predictions_array, actual_values_array, model_name, results_directory, color_name=plot_color, purpose=purpose, add_help_line=True)


    if create_kde:
        # Create KDE plot
        plotting.plot_kde(output_columns, predictions_array, actual_values_array, model_name, results_directory, color_name=plot_color, purpose=purpose)

def apply_model_to_data_transformer_encoder_decoder(model, model_name, results_directory, dataset_path, plot_color, purpose, input_columns, output_columns, batch_size, window_size, window_overlap_size, create_plots=True, create_kde=True):
    test_loader, test_scalers = commons.load_test_data(dataset_path, input_columns, output_columns, batch_size, window_size, window_overlap_size)

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model.to(device)

    # Evaluate the model with test data
    predictions_array, actual_values_array = commons.evaluate_model_get_predictions_and_actuals_transformer_encoder_decoder(model, test_loader, device, batch_size)

    # Calculate metrics for each output parameter and show them
    metrics.calculate_and_show_metrics(results_directory, model_name, purpose, output_columns, predictions_array, actual_values_array)


    if create_plots:
        # Denormalize and plot results for each parameter
        plotting.denorm_and_plot_predicted_actual(output_columns, test_scalers, predictions_array, actual_values_array, model_name, results_directory, color_name=plot_color, purpose=purpose, add_help_line=True)


    if create_kde:
        # Create KDE plot
        plotting.plot_kde(output_columns, predictions_array, actual_values_array, model_name, results_directory, color_name=plot_color, purpose=purpose)
