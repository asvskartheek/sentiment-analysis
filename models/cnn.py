import torch
from torch import nn
from torch.nn import functional as F

from models.bare import Bare

class CNNClassifier(Bare):

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)

        self.embedding = nn.Embedding(self.hparams.vocab_size, self.hparams.embed_dim,
                                      padding_idx=self.hparams.padding_idx)
        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=self.hparams.conv_out_channels,
                                kernel_size=(self.hparams.filter_size, self.hparams.embed_dim))
        self.dropout = nn.Dropout(self.hparams.dropout_rate)
        self.fc = nn.Linear(self.hparams.conv_out_channels, 1)

    def forward(self, example):
        text, text_lens = example
        # text: [max_len, b]
        embedded = self.embedding(text)
        # embedded: [max_len * b * e]
        embedded = embedded.permute(1, 0, 2).unsqueeze(1)
        # embedded: [b * 1 * max_len * e]
        conved = F.relu(self.conv_0(embedded).squeeze(3))
        # conved: [b * out_channels * (max_len - filter_size +1)]
        pooled = F.max_pool1d(conved, conved.shape[2])
        # pooled: [b * out_channels * 1]
        dropped_pooled = self.dropout(pooled.squeeze(2))
        # dropped_pooled: [b * out_channels]
        output = self.fc(dropped_pooled)
        # output: [b * o]

        return output
