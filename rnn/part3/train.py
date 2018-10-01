# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse
import pickle

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import reduce

from dataset import TextDataset
from model import TextGenerationModel

from utils import one_hot, sample_output, string_from_one_hot

################################################################################


def sample_model(model, vocab_size, device=torch.device('cpu'), temp=1.0, hidden_states=None, n=30, prev_char=None):

    if n == 0:
        return prev_char

    # initialize sequence with random character
    if prev_char is None:
        # random character seed
        prev_char = torch.empty(1, 1).random_(0, vocab_size - 1).type(torch.long)
        # convert to one-hot
        prev_char = one_hot(prev_char, vocab_size).to(device)

    output, hidden_states = model(prev_char, hidden_states)
    output = output[:, -1, :] # last prediction

    # Sample next character from softmax
    next_char = sample_output(output, temperature=temp)
    next_char = one_hot(next_char, vocab_size)

    # concat the recursive predictions to currect character
    future = sample_model(model, vocab_size, temp=temp, hidden_states=hidden_states, n=n-1, prev_char=next_char)
    encoded_text = torch.cat([next_char, future], dim=1)

    return encoded_text

def train(config):

    # Initialize the device which to run the model on
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize the dataset and data loader (note the +1)
    abs_path = os.path.abspath(config.txt_file)
    dataset = TextDataset(abs_path, config.seq_length)

    with open('./assets/dataset_seinfeld.pkl', 'wb+') as f:
        pickle.dump(dataset, f)

    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size=config.batch_size,
                                seq_length=config.seq_length,
                                vocabulary_size=dataset.vocab_size,
                                lstm_num_hidden=config.lstm_num_hidden,
                                lstm_num_layers=config.lstm_num_layers,
                                device=device)


    experiment_label = "{}_".format(datetime.now().strftime("%Y-%m-%d %H:%M"))
    for key, value in vars(config).items():
        experiment_label += "{}={}_".format(key, value)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    losses = []
    accuracies = []
    # TODO: configure learning rate scheduler

    for epoch in range(1, config.epochs + 1):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            X = torch.stack(batch_inputs, dim=1)
            X = one_hot(X, dataset.vocab_size)
            Y = torch.stack(batch_targets, dim=1)
            X, Y = X.to(device), Y.to(device)

            # forward pass
            outputs, _ = model(X)

            # compute training metrics
            loss = criterion(outputs.transpose(2, 1), Y)
            predictions = torch.argmax(nn.functional.softmax(outputs, dim=2), dim=2)
            accuracy = (Y == predictions).sum().item() / reduce(lambda x,y: x*y, Y.size())

            losses.append(loss.cpu().item())
            accuracies.append(accuracy)

            # backward pass
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # clip gradients to prevent them form exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), epoch*step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if step % 1000 == 0:
                torch.save(model, './models/seinfeld.pt')

                with open('./results/losses', 'wb+') as f:
                    pickle.dump(losses, f)

                with open('./results/accuracies', 'wb+') as f:
                    pickle.dump(accuracies, f)

        print(' *** End of epoch ***')
        text = sample_model(model=model, vocab_size=dataset.vocab_size, device=device, temp=1)
        generated = string_from_one_hot(text, dataset)

        with open('./results/generated-1-seinfeld.txt', 'a') as f:
            f.write('epoch: {} - '.format(epoch) + generated)


    print('Save final model')
    torch.save(model, './models/final_model.pt')

    with open('./results/losses', 'wb+') as f:
        pickle.dump(losses, f)

    with open('./results/accuracies', 'wb+') as f:
        pickle.dump(accuracies, f)

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6), help='Number of training steps')
    parser.add_argument('--epochs', type=int, default=int(10), help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--temperature', type=float, default=1.0, help='Energy of the probability distribution')
    parser.add_argument('--generate_n', type=int, default=30, help='Length of string to generate')

    config = parser.parse_args()

    # Train the model
    print('Train model')
    train(config)

    # print("-"*20)
    # print('generate Sentences')
    #
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = torch.load('./models/final_model.pt')
    #
    # with open('./assets/dataset.pkl', 'rb') as f:
    #     dataset = pickle.load(f)
    #
    # text = sample_model(model=model, vocab_size=dataset.vocab_size, device=device, temp=1)
    # generated = string_from_one_hot(text, dataset)
    #
    # with open('./results/generated.txt', 'a') as f:
    #     f.write(generated)
