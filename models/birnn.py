import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.bare import Bare


class BiLSTMClassifier(Bare):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)

        self.embedding = nn.Embedding(
            self.hparams.vocab_size,
            self.hparams.embed_dim,
            padding_idx=self.hparams.padding_idx,
        )
        self.dropout = nn.Dropout(self.hparams.dropout_rate)
        self.rnn = nn.LSTM(
            self.hparams.embed_dim,
            self.hparams.hidden_dim,
            num_layers=self.hparams.num_layers,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * self.hparams.hidden_dim, 1)

    def forward(self, example) -> torch.Tensor:
        text, text_lens = example
        # text: [max_len * b]
        embedded = F.relu(self.embedding(text))
        # embedded: [max_len * b * e]
        dropped_embedded = self.dropout(embedded)
        # dropped_embeds: [max_len * b * e]
        packed_sequence = pack_padded_sequence(
            dropped_embedded, text_lens, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.rnn(packed_sequence)
        # hidden, cell: [(num_layers*2) * b * h]
        rnn_output, rnn_output_lens = pad_packed_sequence(packed_output)
        # rnn_output: [max_len * b * (2*h)]
        hidden = hidden.view(self.hparams.num_layers, 2, -1, self.hparams.hidden_dim)
        # hidden: [num_layers * 2 * b * h]
        hidden = hidden[-1]
        # hidden: [2 * b * h]
        hidden_forward, hidden_backward = hidden[0], hidden[1]
        # hidden_forward, hidden_backward: [b * h]
        hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        # hidden: [b * (2*h)]
        dropped_hidden = self.dropout(hidden)
        # dropped_hidden: [b * (2*h)]
        output = self.fc(dropped_hidden)
        # output: [b * o]

        return output
