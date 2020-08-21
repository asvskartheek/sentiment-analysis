import argparse

import torch

from models import *

parser = argparse.ArgumentParser()

# Add all the hyperparameters needed for your model
parser.add_argument("--vocab_size", type=int, default=1000, help="Vocab size")
parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimensions")
parser.add_argument("--padding_idx", type=int, default=1, help="Just for the test nvm")
parser.add_argument(
    "--conv_out_channels", type=int, default=100, help="Conv layer out channels"
)
parser.add_argument(
    "--filter_size", type=int, default=2, help="Default is bigram, set to n-gram"
)
parser.add_argument("--dropout_rate", type=float, default=0.5, help="dropout rate")
#

parser.add_argument(
    "--seed", type=int, default=69, help="seed value for reproducable results"
)

hparams = parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(hparams.seed)
    # torch.backends.cudnn.deterministic = True

    max_len = 20
    b = 8
    new_text = torch.randint(size=(max_len, b), high=hparams.vocab_size)
    n_text_len = torch.randint(size=(max_len, b), high=60)
    example = (new_text, n_text_len)

    ## Line to be changed for model
    model = CNNClassifier(hparams)
    ####

    output = model(example)
    assert output.shape == (b, 1)
    print("Noice! Model complies with the previous designs.")
