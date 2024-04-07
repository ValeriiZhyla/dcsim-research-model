import os

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


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
        plt.title(f'{label} [{model_name}]', fontweight='bold', pad=20, fontsize=16)
        plot.set_xlabel(plot.get_xlabel(), fontdict={'weight': 'bold'}, labelpad=15, fontsize=16),
        plot.set_ylabel(plot.get_ylabel(), fontdict={'weight': 'bold'}, labelpad=15, fontsize=16)

        plot.tick_params(axis='both', which='major', labelsize=14)

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

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_name, markersize=10, alpha=1, label='Model Predictions'),
            Line2D([0], [0], color='#808080', alpha=0.75, label='Ideal Prediction')
        ]
        plt.legend(handles=legend_elements, fontsize='x-large')

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
        plt.title(f'{label} [{model_name}]', fontweight='bold', fontsize=16)
        plt.xlabel('Values', fontweight='bold', fontsize=16)
        plt.ylabel('Density', fontweight='bold', fontsize=16)

        plt.tick_params(axis='both', which='major', labelsize=14)

        # Add legend
        plt.legend(fontsize='x-large')

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
