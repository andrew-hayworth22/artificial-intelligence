import numpy as np
from sklearn.preprocessing import StandardScaler


# ─── Helpers ──────────────────────────────────────────────────────────────────

def calculate_error(actual: np.ndarray, expected: np.ndarray) -> int:
    """Sum of differences between actual and expected binary vectors."""
    return sum(abs(e - a) for a, e in zip(actual, expected))

def vector_to_group(vector: np.ndarray) -> int:
    """Convert a one-hot output vector to a group index."""
    return int(np.argmax(vector))

# ─── Neural Network ───────────────────────────────────────────────────────────

class NN:
    def __init__(self, config: dict):
        self._input_size: int        = config['input_size']
        self._output_size: int       = config['output_size']
        self._hidden_layers: int     = config['hidden_layers']
        self._hidden_layer_size: int = config['hidden_layer_size']
        self._cycles: int            = config['cycles']
        self._fire_threshold: float  = config['fire_threshold']
        self._error_threshold: float = config['error_threshold']
        self._target_margin: float   = config['target_margin']
        self._learning_rate: float   = config['learning_rate']
        self._seed: int              = config['seed']

        self._hidden_bias: list[float] = [config['bias']] * self._hidden_layer_size
        self._output_bias: list[float] = [config['bias']] * self._output_size

        self._scaler  = StandardScaler()
        self._weights = self._initialize_weights()

        print(
            f"Neural Network initialized — "
            f"input: {self._input_size}, "
            f"hidden: {self._hidden_layers}x{self._hidden_layer_size}, "
            f"output: {self._output_size}"
        )

    def train(self, data: np.ndarray, groups: np.ndarray) -> None:
        """Train neural network weight matrices on the provided dataset."""
        print("Training model...")

        # Normalize dataset
        data = self._scaler.fit_transform(data)

        # Cycle over dataset and process each sample
        for cycle in range(self._cycles):
            for idx, (datapoint, group) in enumerate(zip(data, groups)):
                print(
                    f"\rCycle {cycle + 1}/{self._cycles}: sample {idx + 1}/{len(data)}",
                    end='', flush=True
                )
                self._train_sample(datapoint, group)

        print("\nTraining complete.")

    def guess(self, datapoints: np.ndarray) -> list[int]:
        """Predict groups for the provided dataset."""
        # Normalize incoming dataset
        datapoints = self._scaler.transform(datapoints)

        # Return predicted groups for each datapoint
        return [
            vector_to_group(self._forward(np.array(dp))[0][-1])
            for dp in datapoints
        ]

    # ─── Training ─────────────────────────────────────────────────────────────

    def _train_sample(self, datapoint: np.ndarray, group: int) -> None:
        """Run datapoint through neural network and backpropagate error if it exceeds a threshold."""
        # Pass datapoint through network and calculate error
        activations, pre_activations = self._forward(np.array(datapoint))
        expected = self._one_hot(group)
        error    = calculate_error(activations[-1], np.array(expected))

        # If the error exceeds threshold, backpropagate to adjust weights
        if error > self._error_threshold:
            self._backpropagate(activations, pre_activations, expected)

    def _backpropagate(
            self,
            activations: list[np.ndarray],
            pre_activations: list[np.ndarray],
            expected: list[int]
    ) -> None:
        """
        Backpropagate error through the network and update weights.
        Uses pre-activation values to reduce data-loss.
        Uses tanh-derivative to reduce vanishing/exploding gradients.
        """
        output_pre  = np.array(pre_activations[-1])
        expected_arr = np.array(expected)

        # Establish a target for the output layer and calculate error
        target = np.where(expected_arr == 1,
            self._fire_threshold + self._target_margin,
            self._fire_threshold - self._target_margin
        )
        output_error = target - output_pre

        # Skip if already close enough
        if np.all(np.abs(output_error) < self._error_threshold):
            return

        delta = output_error

        # Iterate backwards through layers and update weights
        for layer_idx in range(len(self._weights) - 1, -1, -1):
            layer_input   = np.array(activations[layer_idx])
            weight_update = self._learning_rate * np.outer(delta, layer_input)
            self._weights[layer_idx] += weight_update

            if layer_idx > 0:
                delta = self._weights[layer_idx].T @ delta
                delta = delta * (1 - np.tanh(pre_activations[layer_idx]) ** 2)
                delta = np.clip(delta, -1.0, 1.0)

    def _forward(
            self, starting_vector: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Run a forward pass through the network.
        Returns activations and pre-activation values at every layer.
        Hidden layers use tanh activation function.
        Output layer uses threshold_fire for binary classification.
        """
        activations = [starting_vector]
        pre_activations = [starting_vector]
        v = np.array(starting_vector)

        for i, weight_matrix in enumerate(self._weights):
            bias  = np.array(self._get_bias(i))
            v_in  = weight_matrix @ v + bias
            pre_activations.append(v_in)

            is_output = (i == len(self._weights) - 1)
            v = np.array(self._threshold_fire(v_in)) if is_output else np.tanh(v_in)
            activations.append(v)

        return activations, pre_activations

    def _threshold_fire(self, vector: np.ndarray) -> list[int]:
        """Binary step activation— fires if value exceeds fire_threshold."""
        return [1 if v > self._fire_threshold else 0 for v in vector]

    # ─── Initialization ───────────────────────────────────────────────────────

    def _initialize_weights(self) -> list[np.ndarray]:
        """
        Initialize weight matrices with random values in [-1, 1].
        """
        rng = np.random.default_rng(self._seed)

        weights = [rng.uniform(-1.0, 1.0, (self._hidden_layer_size, self._input_size))]

        for _ in range(self._hidden_layers):
            weights.append(rng.uniform(-1.0, 1.0, (self._hidden_layer_size, self._hidden_layer_size)))

        weights.append(rng.uniform(-1.0, 1.0, (self._output_size, self._hidden_layer_size)))

        return weights

    # ─── Utilities ────────────────────────────────────────────────────────────

    def _one_hot(self, group: int) -> list[int]:
        """Convert a group index to a one-hot encoded vector."""
        vector        = [0] * self._output_size
        vector[group] = 1
        return vector

    def _get_bias(self, layer_index: int) -> list[float]:
        """Return the bias vector for a given layer index."""
        return self._output_bias if layer_index == len(self._weights) - 1 else self._hidden_bias