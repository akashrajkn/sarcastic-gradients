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

from utils import one_hot_encoding, sample_output, string_from_one_hot, sample_sentence

################################################################################


def train(config):
    dataset     = TextDataset(os.path.abspath(config.txt_file), config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Save dataset for future use
    with open('./assets/dataset_x.pkl', 'wb+') as f:
        pickle.dump(dataset, f)

    model       = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden,
                                      config.lstm_num_layers, device)
    criterion   = nn.CrossEntropyLoss()
    optimizer   = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    losses      = []
    accuracies  = []

    # Start training
    for epoch in range(config.epochs):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            x_inp    = one_hot_encoding(torch.stack(batch_inputs, dim=1), dataset.vocab_size).to(device)
            labels   = torch.stack(batch_targets, dim=1).to(device)

            out, _   = model(x_inp)
            loss     = criterion(out.transpose(2, 1), labels)
            preds    = torch.argmax(nn.functional.softmax(out, dim=2), dim=2)
            accuracy = (labels == preds).sum().item() / reduce(lambda x,y: x*y, labels.size())

            model.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

            # Save losses and accuracies
            losses.append(loss.cpu().item())
            accuracies.append(accuracy)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%labels-%m-%d %H:%M"), epoch*step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if step % 1000 == 0:
                torch.save(model, './models/model.pt')
                with open('./results/losses', 'wb+') as f:
                    pickle.dump(losses, f)
                with open('./results/accuracies', 'wb+') as f:
                    pickle.dump(accuracies, f)
            break

        print(' *** End of epoch ***')
        text = sample_sentence(model=model, vocabulary_size=dataset.vocab_size, device=device, temperature=1)
        generated = string_from_one_hot(text, dataset)

        with open('./results/generated-1.txt', 'a') as f:
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
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')
    parser.add_argument('--epochs', type=int, default=int(10), help='Number of training steps')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
