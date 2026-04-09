import itertools
import json


class HMM:
    def __init__(self, config_file):
        with open(config_file, 'r') as cfg:
            config = json.load(cfg)

        self._states: list[str] = config['states']
        self._emissions: list[str] = config['emissions']
        self._initial_probabilities: dict = config['initial_probabilities']
        self._transition_matrix: dict = config['transition_matrix']
        self._emission_matrix: dict = config['emission_matrix']
        self._min_threshold: float = config['min_threshold']

        self._print_config()

    def predict(self, sequence: str) -> None:
        """Predicts the most likely sequence of states based on emissions"""
        # Parse and validate the sequence of emissions
        emissions, valid = self._parse_emission_sequence(sequence)
        if not valid:
            return

        # Generate all possible sequences of states
        state_sets = []
        for emission in emissions:
            state_sets.append(self._get_possible_states(emission))

        possible_paths = list(itertools.product(*state_sets))
        probable_paths = []
        for path in possible_paths:
            if self._validate_state_sequence(path):
                probable_paths.append(path)

        path_probabilities: list[tuple[list[str], float]] = []
        for path in probable_paths:
            path_probabilities.append((path, self._calculate_probability(emissions, path)))

        path_probabilities.sort(key=lambda x: x[1], reverse=True)

        _print_probabilities(path_probabilities)

    def _parse_emission_sequence(self, sequence: str) -> tuple[list[str], bool]:
        """Parses and validates a sequence of emissions"""
        emissions = str.split(sequence, ' ')
        for emission in emissions:
            if emission not in self._emissions:
                print(f"Unknown emission: {emission}")
                return [], False
        return emissions, True

    #
    def _get_possible_states(self, emission: str) -> list[str]:
        """Determines states that could have created an emission"""
        states: list[str] = []
        for state in self._states:
            if abs(self._emission_matrix[state][emission]) >= self._min_threshold:
                states.append(state)
        return states

    def _validate_state_sequence(self, sequence: list[str]) -> bool:
        """Validates whether a state transition sequence is possible or not"""
        # If the first state is not valid, the sequence is not valid
        if abs(self._initial_probabilities[sequence[0]]) < self._min_threshold:
            return False

        # If any state transition is not valid, the sequence is not valid
        for i in range(len(sequence) - 1):
            if abs(self._transition_matrix[sequence[i]][sequence[i + 1]]) < self._min_threshold:
                return False

        return True

    def _calculate_probability(self, emission_sequence: list[str], state_sequence: list[str]) -> float:
        """Calculates the probability of a sequence of states/emissions"""
        # Start with the initial probability and its emission
        state = state_sequence[0]
        emission = emission_sequence[0]
        probability = self._initial_probabilities[state] * self._emission_matrix[state][emission]

        # Apply the transition and emission probabilities
        for i in range(len(state_sequence) - 1):
            probability *= self._transition_matrix[state_sequence[i]][state_sequence[i + 1]]
            probability *= self._emission_matrix[state_sequence[i + 1]][emission_sequence[i + 1]]
        return probability

    def _print_config(self) -> None:
        """Prints the configuration of the HMM"""
        print("HMM Configuration:")
        print("States:", self._states)
        print("Emissions:", self._emissions)
        print("Initial Probabilities:", self._initial_probabilities)
        print("Transition Matrix:", self._transition_matrix)
        print("Emission Matrix:", self._emission_matrix)
        print("Minimum Threshold:", self._min_threshold)

def _print_probabilities(probabilities: list[tuple[list[str], float]]) -> None:
    """Pretty prints the probabilities of possible sequences"""
    for path, probability in probabilities:
        print(f"Path: {path}, Probability: {probability}")