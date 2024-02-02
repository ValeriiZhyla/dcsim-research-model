import lstm_training
from training import commons
import lstm_interpolation
import lstm_extrapolation_x5


class Hyperparameters:
    epochs = 0
    window_size = 0
    window_overlap = 0
    batch_size = 0
    hidden_layers = 0

    def __init__(self, epochs, window_size, window_overlap, batch_size, hidden_layers):
        self.epochs = epochs
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers


hyperparameter_combinations = [
    Hyperparameters(epochs=10, window_size=200, window_overlap=100, batch_size=128, hidden_layers=8),  # test
    Hyperparameters(epochs=200, window_size=200, window_overlap=100, batch_size=128, hidden_layers=8),  # less layers
    Hyperparameters(epochs=200, window_size=200, window_overlap=100, batch_size=128, hidden_layers=50),  # medium layers
    Hyperparameters(epochs=200, window_size=200, window_overlap=100, batch_size=128, hidden_layers=125),  # more layers # good results
    Hyperparameters(epochs=200, window_size=500, window_overlap=250, batch_size=128, hidden_layers=125),  # more layers, larger window
    Hyperparameters(epochs=200, window_size=1000, window_overlap=500, batch_size=128, hidden_layers=125),  # more layers, giant window

    Hyperparameters(epochs=200, window_size=100, window_overlap=50, batch_size=128, hidden_layers=125),  # more layers, smaller window
    Hyperparameters(epochs=200, window_size=50, window_overlap=40, batch_size=128, hidden_layers=125),  # more layers, smaller window

    Hyperparameters(epochs=50, window_size=100, window_overlap=80, batch_size=128, hidden_layers=50),  # Rolling window, medium layers
    Hyperparameters(epochs=50, window_size=100, window_overlap=80, batch_size=128, hidden_layers=125),  # Rolling window, more layers

    Hyperparameters(epochs=200, window_size=50, window_overlap=0, batch_size=128, hidden_layers=125),  # no overlap

]

for hp in hyperparameter_combinations:
    if commons.directory_name_with_hyperparameters_already_exists(lstm_training.model_name, hp.epochs, hp.window_size, hp.window_overlap, hp.batch_size, hp.hidden_layers):
        continue
    else:
        # Train
        model: lstm_training.BiLSTMModel = lstm_training.train_and_evaluate_model(num_epochs=hp.epochs, window_size=hp.window_size, window_overlap=hp.window_overlap,
                                                                                  batch_size=hp.batch_size, hidden_layers=hp.hidden_layers)
        # Save
        directory: str = commons.save_model_and_get_directory(model, lstm_training.model_name, epochs=hp.epochs, window_size=hp.window_size, window_overlap=hp.window_overlap,
                                                              batch_size=hp.batch_size, hidden_layers=hp.hidden_layers)
        print(f"Model trained and saved in {directory}")

        # Evaluate: Interpolation
        lstm_interpolation.evaluate(model, lstm_training.model_name, directory, hp.batch_size, hp.window_size, hp.window_overlap)
        print(f"Model evaluated: interpolation")

        # Evaluate: Extrapolation
        lstm_extrapolation_x5.evaluate(model, lstm_training.model_name, directory, hp.batch_size, hp.window_size, hp.window_overlap)
        print(f"Model evaluated: extrapolation")
        print(f"Model {directory} processed")
        print("===============================")
