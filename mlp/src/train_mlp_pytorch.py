"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import pickle

import torch
import torch.nn as nn

from mlp_pytorch import MLP
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


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    predicted_labels = torch.argmax(predictions, dim=1)
    actual_labels    = torch.argmax(targets, dim=1)
    accuracy         = torch.sum(predicted_labels == actual_labels).to(torch.float32) / targets.shape[0]

    return accuracy


def next_batch_in_tensors(obj, batch_size, device):
    """
    DataSet object returns np arrays. This function converts the batch into tensors

    :param obj       : DataSet object
    :param batch_size: Batch size
    :param device    : cpu or gpu
    :return          : batch and labels as tensors
    """
    batch, labels = obj.next_batch(batch_size)

    # convert batch into tensor
    batch_tensor  = torch.tensor(batch.reshape((batch_size, -1)), dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    return batch_tensor, labels_tensor


def train():
    """
    Performs training and evaluation of MLP model.
    """

    # DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    # Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    # Preparation for training
    print('- Init parameters')
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data            = cifar10_utils.get_cifar10(FLAGS.data_dir)
    train_data      = data['train']
    test_data       = data['test']
    w, h, d         = train_data.images[0].shape
    n_classes       = train_data.labels[0].shape[0]

    criterion       = nn.CrossEntropyLoss()
    model           = MLP(w * h * d, dnn_hidden_units, n_classes).to(device)
    optimizer       = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate)

    train_losses    = []
    test_losses     = []
    test_accuracies = []

    # Train
    print('- Start Training')
    for step in range(FLAGS.max_steps):
        x_batch, x_labels = next_batch_in_tensors(train_data, FLAGS.batch_size, device)

        optimizer.zero_grad()
        out  = model(x_batch)
        loss = criterion(out, x_labels.argmax(dim=1))
        loss.backward()
        optimizer.step()

        train_losses.append(loss.data[0].item())#.cpu().numpy())


        if (step % FLAGS.eval_freq == 0) or (step == FLAGS.max_steps - 1):
            # Test current
            test_x, test_labels = next_batch_in_tensors(test_data, test_data.num_examples, device)

            out_test  = model(test_x)
            loss_test = criterion(out_test, test_labels.argmax(dim=1))
            acc       = accuracy(out_test, test_labels)

            test_losses.append(loss_test.data[0].item())#.cpu().numpy())
            test_accuracies.append(acc.item())#.numpy())

        # if step % 10 == 0:
        #     print('   Step: {}, Train Loss: {}'.format(str(step), str(loss.data[0])))
        #     print('             Test Loss:  {}'.format(str(loss_test.data[0])))

    # Save stuff
    filename = 'steps-{}_layers-{}_lr-{}_bs-{}'.format(FLAGS.max_steps, FLAGS.dnn_hidden_units,
                                                       FLAGS.learning_rate, FLAGS.batch_size)

    filepath = '../models/{}'.format(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    torch.save(model.state_dict(), '{}/model_{}.pt'.format(filepath, filename))

    with open('{}/train_loss_{}'.format(filepath, filename), 'wb+') as f:
        pickle.dump(train_losses, f)

    with open('{}/test_loss_{}'.format(filepath, filename), 'wb+') as f:
        pickle.dump(test_losses, f)

    with open('{}/accuracies_{}'.format(filepath, filename), 'wb+') as f:
        pickle.dump(test_accuracies, f)

    print(test_accuracies[-1])

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
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
