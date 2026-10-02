# Neural Network from Scratch

A neural network of arbitrary depth, built from scratch for a Machine Learning course (DATA 471) as a pair programming project.

## Overview

My partner and I implemented the network using only `numpy` and `argparse`. Every function and helper needed to train and evaluate the network is hand-written in a Python file, [NeuralNetwork.py](NeuralNetwork.py).

Everything is set by the user on the command line and read with `argparse`:

- hyperparameters
- whether the problem is classification or regression
- whether verbose mode is on or off
- how often performance is printed to the terminal

## Features

- Any number of hidden layers and hidden units
- Hidden-layer activations: sigmoid, ReLU, or tanh
- Classification (softmax output, scored by accuracy) and regression (linear output, scored by mean squared error)
- Minibatch gradient descent with backpropagation, shuffling the data every epoch
- Periodic reporting of dev-set performance
- Optional verbose mode that saves parameters, gradients, and minibatches to `.npz` files

## Requirements

- Python 3
- NumPy

```bash
pip install numpy
```

## Usage

```bash
python NeuralNetwork.py \
  -train_feat dataset.train_features.txt \
  -train_target dataset.train_targets.txt \
  -dev_feat dataset.dev_features.txt \
  -dev_target dataset.dev_targets.txt \
  -nunits 16 \
  -nlayers 2 \
  -hidden_act relu \
  -type C \
  -output_dim 3 \
  -total_updates 5000 \
  -learnrate 0.1 \
  -init_range 0.1 \
  -mb 32 \
  -report_freq 100
```

### Arguments

| Flag | Description |
| --- | --- |
| `-train_feat` | Training features file |
| `-train_target` | Training targets file |
| `-dev_feat` | Dev features file |
| `-dev_target` | Dev targets file |
| `-nunits` | Number of units in each hidden layer |
| `-nlayers` | Number of hidden layers |
| `-hidden_act` | Hidden-layer activation: `sig`, `relu`, or `tanh` (the default for any other value) |
| `-type` | Problem type: `C` for classification; any other value means regression |
| `-output_dim` | Output dimension: the number of classes for classification, or the target dimension for regression |
| `-total_updates` | Total number of minibatch updates to run |
| `-learnrate` | Learning rate |
| `-init_range` | Weights and biases start as random values drawn uniformly from `[-init_range, init_range]` |
| `-mb` | Minibatch size (capped at the size of the training set) |
| `-report_freq` | Print dev-set performance every this many updates |
| `-v` | *(Optional)* Directory for verbose output. Setting it turns verbose mode on |

### Output

Progress is printed to the terminal every `report_freq` updates:

```
Epoch 0003 UPDATE 000500: dev=0.912
```

For classification, `dev` is accuracy. For regression, it is mean squared error. In verbose mode, the minibatch score is printed as well (`minibatch=...`), and these files are written to the verbose directory:

- `params_XXXXXX.npz`: weights and biases
- `gradients_XXXXXX.npz`: weight and bias gradients
- `minibatch_XXXXXX.npz`: features and targets of the minibatch

## Data

Data files are plain text that `numpy.loadtxt` can read, with one example per row. The data used is described by the following table:

| Dataset | Train examples | Dev examples | C (output dim) | D (features) |
| --- | --- | --- | --- | --- |
| dataset1 | 22528 | 2482 | 10 | 86 |
| dataset2 | 119 | 17 | 3 | 4 |
| dataset3 | 1159 | 137 | 4 | 22 |
| dataset4 | 27158 | 3003 | 2 | 101 |
| dataset5 | 2792 | 341 | 3 | 10 |
| dataset6 | 10000 | 10000 | 2 | 20 |
| dataset7 | 10000 | 10000 | 2 | 20 |
| dataset8 | 2000 | 2000 | 2 | 20 |
| dataset9 | 100 | 100 | 2 | 50 |
| dataset10 | 500 | 500 | 2 | 5 |
| dataset11 | 2000 | 2000 | 2 | 8 |

## Authors

- Cooper Cox
- Anuk Centellas
