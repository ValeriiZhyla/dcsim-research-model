import torch

import training.commons as commons

def apply_model_to_data(model, model_name, dataset_path, plot_color, purpose, input_columns, output_columns, batch_size, window_size, window_overlap_size, create_plots=True, create_kde=True):
    test_loader, test_scalers = commons.load_test_data(dataset_path, input_columns, output_columns, batch_size, window_size, window_overlap_size)

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model.to(device)

    # Evaluate the model with test data
    predictions_array, actual_values_array = commons.evaluate_model_get_predictions_and_actuals(model, test_loader, device)

    # Calculate metrics for each output parameter and show them
    commons.calculate_and_show_metrics(output_columns, predictions_array, actual_values_array)


    if create_plots:
        # Denormalize and plot results for each parameter
        commons.denorm_and_plot_predicted_actual(output_columns, test_scalers, predictions_array, actual_values_array, model_name, color_name=plot_color, purpose=purpose, add_help_line=True)


    if create_kde:
        # Create KDE plot
        commons.plot_kde(output_columns, predictions_array, actual_values_array, model_name, color_name=plot_color, purpose=purpose)
