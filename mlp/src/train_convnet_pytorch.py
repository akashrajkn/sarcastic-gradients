"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
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

from convnet_pytorch import ConvNet
import cifar10_utils


# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
    batch_tensor  = torch.tensor(batch, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.uint8, device=device)

    return batch_tensor, labels_tensor


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
    batch_tensor  = torch.tensor(batch, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.uint8, device=device)

    return batch_tensor, labels_tensor

def train():
    """
    Performs training and evaluation of ConvNet model.
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    # Preparation for training
    print('- Init parameters')
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data            = cifar10_utils.get_cifar10(FLAGS.data_dir)
    train_data      = data['train']
    test_data       = data['test']
    w, h, d         = train_data.images[0].shape
    n_classes       = train_data.labels[0].shape[0]

    criterion       = nn.CrossEntropyLoss()
    model           = ConvNet(w * h * d, n_classes).to(device)
    optimizer       = torch.optim.Adam(model.parameters())

    train_losses    = []
    test_losses     = []
    test_accuracies = []

    # Train
    print('- Start Training')
    for step in range(FLAGS.max_steps):
        x_batch, x_labels = next_batch_in_tensors(train_data, FLAGS.batch_size, device)

        optimizer.zero_grad()
        out  = model.forward(x_batch)
        loss = criterion(out, x_labels.argmax(dim=1))
        loss.backward()
        optimizer.step()

        train_losses.append(loss.data[0].item())

        if (step % FLAGS.eval_freq == 0) or (step == FLAGS.max_steps - 1):
            print('   - step: {}'.format(step))
            # Test current
            test_x, test_labels = next_batch_in_tensors(test_data, FLAGS.batch_size, device)

            out_test  = model(test_x)
            loss_test = criterion(out_test, test_labels.argmax(dim=1))
            acc       = accuracy(out_test, test_labels)

            test_losses.append(loss_test.data[0].item())
            test_accuracies.append(acc.item())

    # Save stuff

    filepath = '../models/cnn/'
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    torch.save(model, '{}model.pt'.format(filepath))

    with open('{}train_loss'.format(filepath), 'wb+') as f:
        pickle.dump(train_losses, f)

    with open('{}test_loss'.format(filepath), 'wb+') as f:
        pickle.dump(test_losses, f)

    with open('{}accuracies'.format(filepath), 'wb+') as f:
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
