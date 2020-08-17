import argparse
import random

import torch
from torchtext import data, datasets

import pytorch_lightning as pl

from models import SimpleRNNClassifier, FastClassifier, BiLSTMClassifier

from utils import generate_bigrams, count_parameters, save_vocab

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--n_gpus', type=int, default=0, help='number of GPUs')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to be used')
parser.add_argument('--valid', type=float, default=0.2, help='fraction of data to validation')

parser.add_argument('--vocab_size', type=int, default=25_000, help='vocabulary size of text field')
parser.add_argument('--pretrained', type=str, default='glove.6B.100d', help='pretrained vectors to be used for '
                                                                            'Embedding layer')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate to avoid overfitting.')
parser.add_argument('--embed_dim', type=int, default=100, help='dimensions of embedding layer')
parser.add_argument('--hidden_dim', type=int, default=256, help='number of hidden dim for RNN layers')
parser.add_argument('--num_layers', type=int, default=1, help='number of RNN layers')
parser.add_argument('--model', type=str, default='fast', help='model architecture to be used for training, '
                                                              'simple|fast|birnn')
parser.add_argument('--debug', type=bool, default=False, help='run the model in fast_dev_run mode')
parser.add_argument('--overfit_test', type=int, default=0, help='number of batches on which overfit test to be ran')
parser.add_argument('--seed', type=int, default=69, help='seed value for reproducable results')

hparams = parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(hparams.seed)
    torch.backends.cudnn.deterministic = True
    TEXT = data.Field(tokenize='spacy', include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float, is_target=True, unk_token=None)

    if hparams.model == 'fast':
        TEXT = data.Field(tokenize='spacy', include_lengths=True, preprocessing=generate_bigrams)

    print("Splitting dataset into train, valid and test..")

    train, test = datasets.IMDB.splits(TEXT, LABEL)
    valid, train = train.split(split_ratio=hparams.valid, random_state=random.seed(hparams.seed))
    print(f'Number of training examples: {len(train)}')
    print(f'Number of validation examples: {len(valid)}')
    print(f'Number of testing examples: {len(test)}')

    print("Building Vocabulary...")
    TEXT.build_vocab(train,
                     max_size=hparams.vocab_size,
                     vectors=hparams.pretrained,
                     unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train)

    hparams.vocab_size = len(TEXT.vocab)
    hparams.padding_idx = TEXT.vocab.stoi[TEXT.pad_token]

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train, valid, test),
        batch_size=hparams.batch_size)

    print('Creating Model...')
    if hparams.model == 'simple':
        model = SimpleRNNClassifier(hparams)
    elif hparams.model == 'birnn':
        model = BiLSTMClassifier(hparams)
    elif hparams.model == 'fast':
        model = FastClassifier(hparams)
    else:
        raise ValueError("The model doesn't exist select among simple | fast | birnn")
    hparams.parameters = count_parameters(model)
    print(f'The model has {hparams.parameters:,} trainable parameters')

    if hparams.debug:
        trainer = pl.Trainer(fast_dev_run=True)
    elif hparams.overfit_test > 0:
        trainer = pl.Trainer(overfit_batches=hparams.overfit_test, max_epochs=hparams.epochs)
    else:
        trainer = pl.Trainer(max_epochs=hparams.epochs, gpus=hparams.n_gpus)

    print('Training...')
    trainer.fit(model, train_dataloader=train_iterator, val_dataloaders=valid_iterator)

    print('Saving Vocab...')
    save_vocab(TEXT.vocab, 'pre_trained/fast/text.pkl')
    save_vocab(LABEL.vocab, 'pre_trained/fast/label.pkl')

    print('Testing...')
    trainer.test(test_dataloaders=test_iterator)
