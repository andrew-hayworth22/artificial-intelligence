import sys
import json

import numpy as np
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt

from lr import LR

if __name__ == '__main__':
    # Get config file to load
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        exit(1)

    config_path = sys.argv[1]

    # Read config file
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Generate dataset
    seed = config['seed'] if config['seed'] != 0 else None
    x, y = datasets.make_regression(
        n_samples=config['datapoints'],
        n_features=1,
        noise=config['noise'],
        random_state=seed
    )

    # Split dataset into training and testing sets
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=config['test_size'], random_state=seed)
    training_data = np.column_stack((x_train, y_train))

    # Train and test linear regression model
    lr = LR(training_data)
    line = lr.get_line()

    predictions = lr.guess(x_test.flatten())

    # Plot dataset
    plt.scatter(x_train, y_train, color='blue', label='Training Data', alpha=0.5)
    plt.scatter(x_test, y_test, color='green', label='Test Data')
    plt.scatter(x_test, predictions, color='red', label='Predicted Values')

    # Plot line of best fit
    plt.axline(xy1=(0, line[1]), slope=line[0], color='black', label='Line of Best Fit')

    # Plot error lines
    for i in range(len(x_test)):
        print(f"({x_test[i]}, {y_test[i]}) -> ({x_test[i]}, {predictions[i]}): Error = {abs(abs(y_test[i]) - abs(predictions[i]))}")
        plt.plot([x_test[i], x_test[i]], [y_test[i], predictions[i]], color='black', linestyle='--', alpha=0.5, label='Error Lines' if i == 0 else '_nolegend_')

    plt.title("Generated Data for Linear Regression")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.show()