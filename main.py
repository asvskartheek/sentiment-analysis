
from torchtext import data, datasets
import torch

import random

SEED = 69

VALID_RATIO = 0.2

VOCAB_SIZE = 25_000
PRETRAINED_VECTORS = "glove.6B.100d"

BATCH_SIZE = 64


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False, dtype=torch.float, is_target=True, unk_token=None)

    train, test = datasets.IMDB.splits(TEXT, LABEL)
    valid, train = train.split(split_ratio=VALID_RATIO, random_state=random.seed(SEED))
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE)

    print(f'Number of training examples: {len(train)}')
    print(f'Number of validation examples: {len(valid)}')
    print(f'Number of testing examples: {len(test)}')

    TEXT.build_vocab(train,
                     max_size=VOCAB_SIZE,
                     vectors=PRETRAINED_VECTORS,
                     unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    DROPOUT_RATE = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    trainer = pl.Trainer(max_epochs=5, gpus=1)
    trainer.fit(model, train_iterator, valid_iterator)
