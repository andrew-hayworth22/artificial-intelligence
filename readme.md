# Artificial Intelligence Project

This repository contains a series of machine learning models written from scratch for learning purposes.

## 1. Hidden Markov Model

A configurable hidden markov model that can be used to predict sequences of states.

### Running the Model
```
    cd 1-hmm
    python3 main.py config/mood.json
```

### Config File Structure

- **states**: List of hidden states in the model
- **emissions**: List of possible emissions in the model
- **initial_probabilities**: List of probabilities that the model starts in each state
- **transition_matrix**: Matrix of probabilities that the model transitions between states
- **emission_matrix**: Matrix of probabilities that the model emits in each state
- **min-threshold**: The minimum probability that a state must have to be considered

## 2. Neural Network

A configurable neural network that can be used to predict data classifications.

### Running the Model
```
    cd 2-nn
    python3 main.py config/blobs.json
```

### Config File Structure

- **dataset**: Configuration values for random dataset generation
  - **type**: Type of dataset to generate
    - Can be "blobs," "moons," or "circles"
  - **seed**: Seed for random dataset generation
    - 0 for random seed
  - **datapoints**: Number of data points to generate
  - **clusters**: Number of clusters to generate
    - Only applies to blobs dataset
  - **features**: Number of features in each data point
    - Only applies to blobs dataset
  - **factor**: Scale factor between inner and outer circles
    - Only applies to circles dataset
  - **noise**: Standard deviation of noise to add to data points
    - Only applies to circles and moons dataset
  - **test_size**: Percentage of data points to use for testing
    - 0.2 for 20% of data points used for testing
- **nn**: Configuration values for neural network
  - **seed**: Seed for random network initialization
    - 0 for random seed
  - **input_size**: Number of input features
  - **hidden_layers**: Number of hidden layers in the network
  - **hidden_size**: Number of neurons in each hidden layer
  - **output_size**: Number of output classes
  - **bias**: Amount of bias to add to each neuron
    - 0 for no bias
  - **learning_rate**: Learning rate for the network
  - **cycles**: Number of training cycles to run
  - **weight_adjustment_max**: Maximum weight adjustments for each cycle
  - **fired_threshold**: Threshold for neuron activation
  - **error_threshold**: Threshold for error
  - **target_margin**: Margin of error for target value
  - **epsilon**: Value close to zero for numerical stability
- **render_graph**: Whether or not to render a 2D graph of the network's training results
  - Only applicable for datasets with 2 input features

## 3. K-Means Clustering

Clustering of data points using the k-means clustering algorithm.

### Running the Model

```
    cd 3-km
    python3 main.py config.json
```

### Config File Structure

- **seed**: Seed for random dataset generation
  - 0 for random seed
- **datapoints**: Number of data points to generate
- **clusters**: Number of clusters to generate
- **clusters_std**: Standard deviation of clusters
- **centroid_shift**: Amount to shift centroids by
- **cycles**: Number of training cycles to run
- **test_size**: Percentage of data points to use for testing
  - 0.2 for 20% of data points used for testing

## 4. Linear Regression

Linear regression of data points.

### Running the Model

```
    cd 4-lr
    python3 main.py config.json
```

### Config File Structure

- **seed**: Seed for random dataset generation
  - 0 for random seed
- **datapoints**: Number of data points to generate
- **noise**: Noise to add to data points
- **test_size**: Percentage of data points to use for testing
  - 0.2 for 20% of data points used for testing

## 5. CNN Photo Classification

Not quite finished, but will be a CNN that can classify photos of cats and houses.