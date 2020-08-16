import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from bare import Bare


class SimpleRNNClassifier(Bare):

    def __init__(self, hparams: dict):
        super().__init__(hparams)

        self.embedding = nn.Embedding(hparams['vocab_size'], hparams['embed_dim'], padding_idx=hparams['padding_idx'])
        self.dropout = nn.Dropout(hparams['dropout_rate'])
        self.rnn = nn.RNN(hparams['embed_dim'], hparams['hidden_dim'])
        self.fc = nn.Linear(hparams['hidden_dim'], hparams['output_dim'])

    def forward(self, example) -> torch.Tensor:
        text, text_lens = example
        # text: [max_len * b]
        embedded = F.relu(self.embedding(text))
        # embedded: [max_len * b * e]
        dropped_embedded = self.dropout(embedded)
        # dropped_embeds: [max_len * b * e]
        packed_sequence = pack_padded_sequence(dropped_embedded,
                                               text_lens,
                                               enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_sequence)
        rnn_output, rnn_output_lens = pad_packed_sequence(
            packed_output)
        # rnn_output: [max_len * b * h]
        # hidden: [1 * b * h]
        dropped_hidden = self.dropout(hidden.squeeze(0))
        # dropped_hidden: [b * h]
        output = self.fc(dropped_hidden)
        # output: [b * o]

        return output
