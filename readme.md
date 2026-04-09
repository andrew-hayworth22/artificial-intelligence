# Hidden Markov Model

## Running the program
Pass in any config file path as an argument to begin:

``python3 main.py config/mood.json``

## Config file
Located in the config directory, config files have data needed to run the hidden markov model and begin predicting state sequences.

The following fields are required in each config JSON file:
- **states**: list of possible model states
- **emissions**: list of possible emissions
- **initial_probabilities**: list of probabilities that the model starts in each state
- **transition_matrix**: matrix of probabilities that the model transitions between states
- **emission_matrix**: matrix of probabilities that the model emits in each state
- **min-threshold**: the minimum probability that a state must have to be considered