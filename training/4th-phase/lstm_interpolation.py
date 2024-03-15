import torch

import apply_model
from lstm_training_main import BiLSTMModel, model_name, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, plot_color
import scenarios

purpose = "interpolation"

MODEL_PATH = "generated-models/default/lstm.pth"
PLOTS_PATH = "generated-models/default/plots"


def evaluate(model, model_name, results_directory, batch_size, window_size, window_overlap, scenario):
    apply_model.apply_model_to_data_with_platform_information(model, model_name, results_directory,
                                                              scenario.test_dataset_path, scenario.nodes_aux_path, scenario.links_aux_path,
                                                              scenarios.input_columns_jobs_numerical, scenarios.input_columns_jobs_categorical,
                                                              scenarios.output_columns_jobs_numerical, scenarios.output_columns_jobs_categorical,
                                                              scenarios.nodes_columns_numerical, scenarios.nodes_columns_categorical,
                                                              scenarios.links_columns_numerical, scenarios.links_columns_categorical,
                                                              plot_color, purpose,
                                                              batch_size, window_size, window_overlap)


if __name__ == '__main__':
    model: BiLSTMModel = torch.load(MODEL_PATH)
    evaluate(model, model_name, PLOTS_PATH, BATCH_SIZE, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, scenarios.main_scenario)
