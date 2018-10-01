import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import reduce


def one_hot(batch, vocab_size):

    X = torch.zeros(batch.shape[0], batch.shape[1], vocab_size, device=batch.device)
    X.scatter_(2, batch[:,:,None], 1)

    return X


def sample_output(out, temperature=1.0):
    '''
    Function takes the out and returns a sample based on the temperature value
    Args:
        @out
        @temperature: value of temperature controls random sampling and greedy sampling
    Return:
        @sample
    '''

    out   = nn.functional.softmax(out, dim=1)
    sample = torch.log(output) / temperature
    sample = torch.exp(sample) / (torch.exp(sample).sum())

    return torch.multinomial(sample, 1)


def string_from_one_hot(sequence, dataset):
    '''
    Converts 1-hot encoding to string
    '''
    char_idxs = sequence.argmax(dim=2).squeeze_(0).cpu().numpy()
    return dataset.convert_to_string(char_idxs)
