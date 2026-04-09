import json
import sys

import numpy as np
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from km import KM

if __name__ == "__main__":
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
    data, groups = datasets.make_blobs(
        n_samples = config['datapoints'],
        centers = config['clusters'],
        cluster_std = config['cluster_std'],
        n_features = 2,
        random_state = seed,
    )
    data_train, data_test, classes_train, classes_test = model_selection.train_test_split(data, groups, test_size=config['test_size'], random_state=seed)

    # Train K-Means model
    km = KM(config, data_train)

    # Test K-Means model
    results = km.guess(data_test)
    outliers = results.count(-1)
    print(f"Final Outliers: {outliers}")

    # Plot results
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=-1, vmax=config['clusters'] - 1)

    fig, ax = plt.subplots()
    ax.set_title(f"K-Means Clustering Results - {config['clusters']} Clusters")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    train_scatter = ax.scatter(data_train[:, 0], data_train[:, 1], c=classes_train, cmap='viridis', s=40, alpha=0.7, label="Training Data", norm=norm)
    test_scatter = ax.scatter(data_test[:, 0], data_test[:, 1], c=results, cmap='viridis', s=150, marker="X", edgecolors='black', linewidth=1.5, label="Test Data", norm=norm)
    fig.colorbar(train_scatter, ax=ax, label="Training Cluster")
    fig.colorbar(test_scatter, ax=ax, label="Test Cluster")
    ax.legend()

    # Plot bounding box and centroids
    bounds = km.get_bounds()
    rect_x = bounds[0]
    rect_y = bounds[2]
    rect_width = bounds[1] - bounds[0]
    rect_height = bounds[3] - bounds[2]
    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                             linewidth=2, edgecolor='r', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    centroids = km.get_centroids()
    for idx in range(len(centroids)):
        centroid = centroids[idx]
        circle = plt.Circle(centroid[0], centroid[1], color=cmap(norm(idx)), fill=False, linewidth=2)
        ax.add_patch(circle)

    plt.show()

