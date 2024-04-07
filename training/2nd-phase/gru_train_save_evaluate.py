import gru_training
from training import commons
import gru_interpolation
import gru_extrapolation_x5


class Hyperparameters:
    epochs = 0
    window_size = 0
    window_overlap = 0
    batch_size = 0
    hidden_size = 0
    layers = 1

    def __init__(self, epochs, window_size, window_overlap, batch_size, hidden_size, layers):
        self.epochs = epochs
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.layers = layers


hyperparameter_combinations = [
    # Hyperparameters(epochs=10, window_size=200, window_overlap=100, batch_size=128, hidden_size=8, layers=1),  # test
    #
    # Hyperparameters(epochs=100, window_size=200, window_overlap=100, batch_size=128, hidden_size=50, layers=1),  # test with windows 200/100
    # Hyperparameters(epochs=100, window_size=100, window_overlap=50, batch_size=128, hidden_size=50, layers=1),  # test with windows 100/50
    # Hyperparameters(epochs=100, window_size=100, window_overlap=90, batch_size=128, hidden_size=50, layers=1),  # test with windows 100/90
    # Hyperparameters(epochs=100, window_size=50, window_overlap=40, batch_size=128, hidden_size=50, layers=1),  # test with windows 50/40
    # Hyperparameters(epochs=100, window_size=100, window_overlap=0, batch_size=128, hidden_size=50, layers=1),  # test with windows 100/0
    # Hyperparameters(epochs=100, window_size=50, window_overlap=0, batch_size=128, hidden_size=50, layers=1),  # test with windows 100/0
    #
    # # Check different hidden sizes
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=128, layers=1),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=96, layers=1),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=64, layers=1),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=32, layers=1),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, layers=1), # <== 16 hidden is the best
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=8, layers=1),
    #
    # # Check different batch sizes
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=256, hidden_size=16, layers=1),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, layers=1), # <== 128 batch size it the best
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=64, hidden_size=16, layers=1),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=32, hidden_size=16, layers=1),
    #
    # # Check different layers number
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, layers=1), # <== 1 layer is the best
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, layers=2),
    # Hyperparameters(epochs=300, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, layers=4),
    #
    # # Check more layers on longer training
    # Hyperparameters(epochs=500, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, layers=4),
    # Hyperparameters(epochs=500, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, layers=6),
    # Hyperparameters(epochs=500, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, layers=8),
    # Hyperparameters(epochs=500, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, layers=10), # 1 layer acts better
    #
    # # Check different window sizes
    # Hyperparameters(epochs=600, window_size=50, window_overlap=0, batch_size=128, hidden_size=16, layers=1), # <== 50 is better
    # Hyperparameters(epochs=600, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, layers=1), # <== 100 is optimal
    # Hyperparameters(epochs=600, window_size=150, window_overlap=0, batch_size=128, hidden_size=16, layers=1),
    # Hyperparameters(epochs=600, window_size=200, window_overlap=0, batch_size=128, hidden_size=16, layers=1),
    #
    # # Check different window overlaps
    # Hyperparameters(epochs=700, window_size=100, window_overlap=0, batch_size=128, hidden_size=16, layers=1), # <== No ovelap is the best
    # Hyperparameters(epochs=700, window_size=100, window_overlap=10, batch_size=128, hidden_size=16, layers=1),
    # Hyperparameters(epochs=700, window_size=100, window_overlap=20, batch_size=128, hidden_size=16, layers=1),
    # Hyperparameters(epochs=700, window_size=100, window_overlap=30, batch_size=128, hidden_size=16, layers=1),
    # Hyperparameters(epochs=700, window_size=100, window_overlap=40, batch_size=128, hidden_size=16, layers=1),
    # Hyperparameters(epochs=700, window_size=100, window_overlap=50, batch_size=128, hidden_size=16, layers=1),
    # # 170 epocs -> convergence
    #
    # # More windows sizes
    # Hyperparameters(epochs=251, window_size=50, window_overlap=0, batch_size=128, hidden_size=16, layers=1),
    # Hyperparameters(epochs=251, window_size=25, window_overlap=0, batch_size=128, hidden_size=16, layers=1), # <== 25
    # Hyperparameters(epochs=251, window_size=10, window_overlap=0, batch_size=128, hidden_size=16, layers=1),

    Hyperparameters(epochs=101, window_size=25, window_overlap=0, batch_size=128, hidden_size=16, layers=1)
    # Best model is Hyperparameters(epochs=251, window_size=25, window_overlap=0, batch_size=128, hidden_size=16, layers=1)

]

for hp in hyperparameter_combinations:
    if commons.directory_name_with_hyperparameters_already_exists(gru_training.model_name, hp.epochs, hp.window_size, hp.window_overlap, hp.batch_size, hp.hidden_size, layers=hp.layers):
        continue
    else:
        # Train
        model: gru_training.BiGRUModel = gru_training.train_and_evaluate_model(num_epochs=hp.epochs, window_size=hp.window_size, window_overlap=hp.window_overlap,
                                                                               batch_size=hp.batch_size, hidden_size=hp.hidden_size, layers=hp.layers)
        # Save
        directory: str = commons.save_model_and_get_directory(model, gru_training.model_name, epochs=hp.epochs, window_size=hp.window_size, window_overlap=hp.window_overlap,
                                                              batch_size=hp.batch_size, hidden_size=hp.hidden_size, layers=hp.layers)
        print(f"Model trained and saved in {directory}")

        # Evaluate: Interpolation
        gru_interpolation.evaluate(model, gru_training.model_name, directory, hp.batch_size, hp.window_size, hp.window_overlap)
        print(f"Model evaluated: interpolation")

        # Evaluate: Extrapolation
        gru_extrapolation_x5.evaluate(model, gru_training.model_name, directory, hp.batch_size, hp.window_size, hp.window_overlap)
        print(f"Model evaluated: extrapolation")
        print(f"Model {directory} processed")
        print("===============================")
