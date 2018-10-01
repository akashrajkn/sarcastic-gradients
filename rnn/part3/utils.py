import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import reduce


def one_hot_encoding(x_batch, vocab_size):
    '''
    Returns the one hot encoding of x_batch
    '''
    X = torch.zeros(x_batch.shape[0], x_batch.shape[1], vocab_size, device=x_batch.device)

    return X.scatter_(2, x_batch[:,:,None], 1)


def sample_output(out, temperature=1.0):
    '''
    Function takes the out and returns a sample based on the temperature value
    Args:
        @out
        @temperature: value of temperature controls random sampling and greedy sampling
    Return:
        @sample
    '''

    out    = nn.functional.softmax(out, dim=1)
    sample = torch.log(out) / temperature
    sample = torch.exp(sample) / (torch.exp(sample).sum())

    return torch.multinomial(sample, 1)


def string_from_one_hot(sequence, dataset):
    '''
    Converts 1-hot encoding to string
    Args:
        @sequence
        @dataset : TextDataset object
    Return:
        @string
    '''
    return dataset.convert_to_string(sequence.argmax(dim=2).squeeze_(0).cpu().numpy())


def sample_sentence(model, vocabulary_size, hidden_states=None, device=torch.device('cpu'), temperature=1.0, seq_len=30, previous=None):
    '''
    Samples a sentence from the model
    Args:
        @model          : LSTM model
        @vocabulary_size
        @temperature           : temperature setting
        @hidden_states
        @seq_len        : number of characters to generate
        @previous      : predicted char
    '''
    if seq_len == 0:
        return previous

    if previous is None:
        previous = one_hot_encoding(torch.empty(1, 1).random_(0, vocabulary_size - 1).type(torch.long), vocabulary_size).to(device)

    out, hidden_states = model(previous, hidden_states)
    out                = out[:, -1, :]
    current            = one_hot_encoding(sample_output(out, temperature=temperature), vocabulary_size)
    predicted          = sample_sentence(model=model, vocabulary_size=vocabulary_size, temperature=temperature,
                                         hidden_states=hidden_states, seq_len=seq_len-1, previous=current)

    return torch.cat([current, predicted], dim=1)

def get_accuracy(labels, preds):
    return (labels == preds).sum().item() / reduce(lambda x, y: x*y, labels.size())
