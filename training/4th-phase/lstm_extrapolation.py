import torch

import apply_model
from lstm_training_main import BiLSTMModel, model_name, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, BATCH_SIZE, plot_color
import scenarios

purpose = "extrapolation"

MODEL_PATH = "generated-models/lstm_1/lstm_1layer_256hs.pth"
PLOTS_PATH = "generated-models/lstm_1/plots_base-17"

# MODEL_PATH = "generated-models/LSTM_4layer_512hs/lstm_base.pth"
# PLOTS_PATH = "generated-models/LSTM_4layer_512hs/plots_base"

#MODEL_PATH = "generated-models/LSTM_4layer_512hs/lstm_tuned_100each_3iter.pth"
#PLOTS_PATH = "generated-models/LSTM_4layer_512hs/plots_tuned-17"

def evaluate(model, model_name, results_directory, batch_size, window_size, window_overlap, scenario, extrapolation_path):
    apply_model.apply_model_to_data_with_platform_information(model, model_name, results_directory,
                                                              extrapolation_path, scenario.nodes_aux_path, scenario.links_aux_path,
                                                              scenarios.input_columns_jobs_numerical, scenarios.input_columns_jobs_categorical,
                                                              scenarios.output_columns_jobs_numerical, scenarios.output_columns_jobs_categorical,
                                                              scenarios.nodes_columns_numerical, scenarios.nodes_columns_categorical,
                                                              scenarios.links_columns_numerical, scenarios.links_columns_categorical,
                                                              plot_color, purpose,
                                                              batch_size, window_size, window_overlap)


if __name__ == '__main__':
    # extrapolation_path = '../../dataset_preparation/4th-phase/0-main-dataset/extrapolation_dataset.csv'

    extrapolation_path = '../../dataset_preparation/4th-phase/extrapolation/17-kit_desy-cache-moved-to-desy-two-more-desy-hosts-and-more-resources-less-hosts-in-kit-and-less-resources-extrapolation/extrapolation_dataset.csv'

    model: BiLSTMModel = torch.load(MODEL_PATH)
    evaluate(model, model_name, PLOTS_PATH, BATCH_SIZE, WINDOW_SIZE, WINDOW_OVERLAP_SIZE, scenarios.main_scenario, extrapolation_path)
