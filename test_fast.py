import random

import torch
from torchtext import data, datasets

import pytorch_lightning as pl

from fast import FastClassifier
from utils import generate_bigrams, count_parameters

SEED = 69
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

hparams = {
    'vocab_size': 25_000,
    'pretrained_vecs': "glove.6B.100d",
    'embedding_dim': 100,
    'output_dim': 1,
    'dropout_rate': 0.5,
    'optimizer': 'adam',
    'learning_rate': 1e-3,
    'valid_ratio': 0.2,
    'batch_size': 8,  # 64 for GPU machines
}

if __name__ == '__main__':
    TEXT = data.Field(tokenize='spacy', include_lengths=True, preprocessing=generate_bigrams)
    LABEL = data.LabelField(dtype=torch.float)

    train, test = datasets.IMDB.splits(TEXT, LABEL)
    valid, train = train.split(split_ratio=hparams['valid_ratio'], random_state=random.seed(SEED))

    TEXT.build_vocab(train,
                     max_size=hparams['vocab_size'],
                     vectors=hparams['pretrained_vecs'],
                     unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train)

    hparams['vocab_size'] = len(TEXT.vocab)
    hparams['padding_idx'] = TEXT.vocab.stoi[TEXT.pad_token]

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train, valid, test),
        batch_size=hparams['batch_size'])

    # hparams['train_loader'] = train_iterator
    # hparams['val_loader'] = valid_iterator
    # hparams['test_loader'] = test_iterator

    model = FastClassifier(hparams)
    model.pre_trained_embeds(TEXT.vocab.vectors, zero_words=[TEXT.vocab.stoi[TEXT.unk_token],
                                                             hparams['padding_idx']])
    print(f'The model has {count_parameters(model):,} trainable parameters')

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, train_dataloader=train_iterator, val_dataloaders=valid_iterator)
    trainer.test(test_dataloaders=test_iterator)
