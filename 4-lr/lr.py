import numpy as np


class LR:
    def __init__(self, data):
        self._data = data
        self._train()

    def _train(self):
        """Calculates the line of best fit for the given data."""

        print("Training linear regression model...")
        # Calculate the mean x and y values
        x = self._data[:, 0]
        y = self._data[:, 1]
        x_avg = np.mean(x)
        y_avg = np.mean(y)

        # Calculate the slope and intercept
        sigma_x = np.sum((x - x_avg) ** 2) / (len(self._data) - 1)
        covariance_xy = np.sum((x - x_avg) * (y - y_avg)) / (len(self._data) - 1)

        self._slope = covariance_xy / sigma_x
        self._intercept = y_avg - self._slope * x_avg
        print(f"Linear regression model trained: y = {self._slope:.2f}x + {self._intercept:.2f}")
        print("-"*30)

    def guess(self, x_values):
        """Returns the predicted y values for the given x values."""
        predictions = []
        for x in x_values:
            y = self._slope * x + self._intercept
            predictions.append(y)
        return predictions


    def get_line(self):
        """Returns the slope and intercept of the line of best fit."""
        return self._slope, self._intercept