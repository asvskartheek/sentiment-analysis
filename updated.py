import pytorch_lightning as pl
import torch
from torchtext import data, datasets
import random

from birnn import BiLSTMClassifier
from utils import count_parameters

SEED = 69
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True


VALID_RATIO = 0.2
MAX_EPOCHS = 5

VOCAB_SIZE = 25_000
PRETRAINED_VECTORS = "glove.6B.100d"

BATCH_SIZE = 8  # 64 for GPU machines
EMBEDDING_DIM = 100
NUM_LAYERS = 2
HIDDEN_DIM = 256
OUTPUT_DIM = 1
DROPOUT_RATE = 0.5

if __name__ == '__main__':
    TEXT = data.Field(lower=True, include_lengths=True)
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
    LABEL.build_vocab(train) # pos: 0, neg: 1

    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

    VOCAB_SIZE = len(TEXT.vocab)

    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    classifier_model = BiLSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, NUM_LAYERS, HIDDEN_DIM,
                                        OUTPUT_DIM, PAD_IDX, DROPOUT_RATE)
    print(f'The model has {count_parameters(classifier_model):,} trainable parameters')

    pretrained_embeddings = TEXT.vocab.vectors
    classifier_model.embedding.weight.data.copy_(pretrained_embeddings)

    # Makes Zero Useless Embeddings
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    classifier_model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    classifier_model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    trainer = pl.Trainer(overfit_batches=0.001)
    trainer.fit(classifier_model, train_iterator, valid_iterator)
