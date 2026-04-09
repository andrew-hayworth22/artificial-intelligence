import json
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import validate_config, generate_dataset
from nn import NN

if __name__ == '__main__':
    # Get config file to load
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        exit(1)

    config_path = sys.argv[1]

    # Read and validate config file
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    validate_config(config)
    config_dataset = config['dataset']
    config_nn = config['nn']

    # Generate dataset
    data_train, data_test, classes_train, classes_test = generate_dataset(config_dataset)

    # Train neural network
    nn = NN(config_nn)
    nn.train(data_train, classes_train)

    # Test neural network
    results = nn.guess(data_test)

    # Evaluate results
    correct = 0
    print('*'*15 + " RESULTS " + '*'*15 + "\n")
    for i in range(len(results)):
        expected = classes_test[i]
        actual = results[i]
        is_correct = expected == actual

        if is_correct:
            correct += 1

        print(f"{'+' if is_correct else '-'} Datapoint {data_test[i]}; Expected: {expected}; Actual: {actual}")

    print("*" * 39)
    print(f"Accuracy: {correct / len(results) * 100:.2f}%")

    # Render graph if enabled
    if config['render_graph']:
        norm = mcolors.Normalize(vmin=0, vmax=config_nn['output_size'] - 1)

        plt.figure(figsize=(8, 6))
        plt.title(f"Neural Network Clustering Results - {config_dataset['type']} Dataset")
        plt.scatter(data_train[:, 0], data_train[:, 1], c=classes_train, cmap='viridis', s=40, alpha=0.7, label="train", norm=norm)
        plt.scatter(data_test[:, 0], data_test[:, 1], c=results, cmap='viridis', s=150, marker="X", edgecolors='black', linewidth=1.5, label="test", norm=norm)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(label="Cluster")
        plt.show()

