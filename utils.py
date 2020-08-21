import torch
from torch import nn


def binary_accuracy(logits, y_true):
    rounded_preds = torch.round(torch.sigmoid(logits))
    correct = (rounded_preds == y_true).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def bce_loss_with_logits(logits, y_true):
    loss_func = nn.BCEWithLogitsLoss()
    return loss_func(logits, y_true)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(" ".join(n_gram))
    return x


def save_vocab(vocab, path):
    import pickle

    output = open(path, "wb")
    pickle.dump(vocab, output)
    output.close()
