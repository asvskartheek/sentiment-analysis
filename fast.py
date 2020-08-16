import torch
from torch import nn
from torch.nn import functional as F

from bare import Bare


class FastClassifier(Bare):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams)
        self.hparams = hparams

        self.embedding = nn.Embedding(hparams['vocab_size'], hparams['embedding_dim'],
                                      padding_idx=hparams['padding_idx'])

        self.fc = nn.Linear(hparams['embedding_dim'], hparams['output_dim'])
        self.dropout = nn.Dropout(hparams['dropout_rate'])

    def forward(self, example):
        text, text_lens = example
        # text: [max_len * b]
        embedded = F.relu(self.embedding(text))
        # embedded: [max_len * b * e]
        dropped_embedded = self.dropout(embedded)
        # dropped_embedded: [max_len * b * e]
        dropped_embedded = dropped_embedded.permute(1, 0, 2)
        # dropped_embedded: [b * max_len * e]
        pooled = F.avg_pool2d(dropped_embedded, (dropped_embedded.shape[1], 1)).squeeze(1)
        # pooled: [b * e]
        output = self.fc(pooled)
        # output: [b * o]

        return output

    def pre_trained_embeds(self, embeddings, zero_words=None):
        self.embedding.weight.data.copy_(embeddings)

        if zero_words is not None:
            for word in zero_words:
                self.embedding.weight.data[word] = torch.zeros(self.hparams['embedding_dim'])
