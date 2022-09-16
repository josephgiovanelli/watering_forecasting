import numpy as np

"""Class implementing the "Persistent System" behaviour.
"""


class PersistentSystem:
    def __init__(self, window_size, y_size):
        self.window_size = window_size
        self.y_size = y_size

    def fit(self, X_train, y_train):
        return self

    def predict(self, dataset):
        predictions = np.ndarray(shape=(dataset.shape[0], self.y_size))

        for i in range(0, predictions.shape[0]):
            for j in range(0, predictions.shape[1]):
                predictions[i][j] = dataset[i][self.window_size * 6 + j]

        return predictions
