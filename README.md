Efficient Machine Learning - The Bird's Nest
## Tejaswini Ramkumar Babu
This repository contains the code for the CS 4803/8803 Efficient ML Final Project, "The Bird’s Nest: Investigating Early-Bird Pruning on Transformers Using Various Pruning Techniques"

## `bert.py`
### Description:
- Trains a BERT model for sequence classification on the SST-2 dataset.
- Uses the EarlyBird class to detect the optimal training point for pruning, potentially reducing unnecessary training iterations.
- Uses Magnitude, Gradient-based, and Undecayed pruning methods to reduce model size and potentially improve inference times.
- Includes custom `Trainer` classes for integrating pruning within the training process.

- To run, do python3/python bert.py
- Adjust pruning rate and replace custom `Trainer' with 'Trainer' for magnitude pruning, 'GradientTrainer' for gradient pruning, and 'UndecayedTrainer' for undecayed pruning.

## `roberta-3.py`
### Description:
- Trains a RoBERTa model for sequence classification on the SST-2 dataset.
- Uses the EarlyBird class to detect the optimal training point for pruning, potentially reducing unnecessary training iterations.
- Uses Magnitude, Gradient-based, and Undecayed pruning methods to reduce model size and potentially improve inference times.
- Includes custom `Trainer` classes for integrating pruning within the training process.

- To run, do python3/python roberta-3.py
- Adjust pruning rate and replace custom `Trainer' with 'Trainer' for magnitude pruning, 'GradientTrainer' for gradient pruning, and 'UndecayedTrainer' for undecayed pruning.


## `gradient.py`
### Description:
- A helper file for gradient pruning.

## `undecayed.py`
### Description:
- A helper file for undecayed pruning.

## `EarlyBird.py`
- The helper file to find the Early Bird for integration with the BERT model (EarlyBird class) and the RoBERTa model (EarlyBirdRoBERTa class)
- Implements a function to prune model parameters using gradient information
- Calculates a cutoff threshold dynamically based on the specified amount of parameters to be pruned
- Directly modifies the parameters of the model based on the calculated mask that signifies which weights survive the pruning process


