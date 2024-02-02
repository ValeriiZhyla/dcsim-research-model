import os

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_and_show_metrics(results_directory, model_name, purpose, output_columns, predictions_array, actual_values_array):
    mse_per_param, mae_per_param, rmse_per_param, r_squared_per_param = calculate_metrics(actual_values_array, predictions_array, output_columns)
    show_and_save_metrics(results_directory, purpose, output_columns, mse_per_param, mae_per_param, rmse_per_param, r_squared_per_param)
    save_metrics_latex(results_directory, model_name, purpose, output_columns, mse_per_param, mae_per_param, rmse_per_param, r_squared_per_param)


def save_metrics_latex(results_directory, model_name, purpose, output_columns, mse_per_param, mae_per_param, rmse_per_param, r_squared_per_param):
    p = 5
    lines_start = [r"\begin{table}[ht!]",
                   r"\centering",
                   r"\caption{" + model_name + ": " + purpose.capitalize() + " accuracy metrics}",
                   r"\label{table:TODO-PHASE-LABEL/" + model_name.lower() + "-" + purpose.lower() + "-metrics}",
                   r"\renewcommand{\arraystretch}{1.25}",
                   r"\begin{tabular}{c|c>{\columncolor{lightgray}}cc>{\columncolor{lightgray}}c}",
                   r"\hline",
                   r"%\rowcolor{white} % Keeps the header row white",
                   r"\textbf{Parameter} & \textbf{MSE} & \textbf{RMSE} & \textbf{MAE} & \textbf{R\textsuperscript{2}} \\",
                   r"\hline"]

    lines_end = [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}"
    ]

    lines_body = []
    for i in range(len(output_columns)):
        name = output_columns[i]
        name = name.replace("_", "\_")
        line = f"{name} & {str(round(mse_per_param[i], p))} & {str(round(rmse_per_param[i], p))} & {str(round(mae_per_param[i], p))} & {str(round(r_squared_per_param[i], p))} \\\\"
        lines_body.append(line)

    lines = lines_start + lines_body + lines_end

    file_name = os.path.join(results_directory, f"metrics_{purpose}_latex.txt")
    with open(file_name, 'w') as file:
        for line in lines:
            file.write(line + '\n')
    print("Latex table saved to {}".format(file_name))


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
