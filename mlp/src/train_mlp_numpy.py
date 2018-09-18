"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import pickle

from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, x_labels):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the x_batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole x_batch

    TODO:
    Implement accuracy computation.
    """

    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(x_labels, axis=1)
    accuracy = np.sum(predicted_labels == actual_labels) / x_labels.shape[0]

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    # Preparation for training
    print('- Init parameters')

    data       = cifar10_utils.get_cifar10(FLAGS.data_dir)
    train_data = data['train']
    test_data  = data['test']
    w, h, d    = train_data.images[0].shape
    n_classes  = train_data.labels[0].shape[0]

    criterion  = CrossEntropyModule()
    model      = MLP(w * h * d, dnn_hidden_units, n_classes)

    train_losses = []
    test_losses = []
    accuracies = []

    print('- Start training')
    for step in range(FLAGS.max_steps):

        x_batch, x_labels = train_data.next_batch(FLAGS.batch_size)
        x                 = x_batch.reshape((FLAGS.batch_size, -1))
        predictions       = model.forward(x)
        gradient          = criterion.backward(predictions, x_labels)

        model.backward(gradient)
        model.step(FLAGS.learning_rate)

        if step % FLAGS.eval_freq == 0 or step == FLAGS.max_steps - 1:
            print('    - Step: {}'.format(step))
            loss      = criterion.forward(predictions, x_labels)
            out_test  = model.forward(test_data.images.reshape(test_data.num_examples, -1))
            test_loss = criterion.forward(out_test, test_data.labels)
            acc       = accuracy(out_test, test_data.labels)

            train_losses.append(loss)
            test_losses.append(test_loss)
            accuracies.append(acc)

    # Save stuff
    print(accuracies[-1])


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing x data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
