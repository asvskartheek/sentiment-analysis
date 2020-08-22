from pathlib import Path

from pytorch_lightning import LightningDataModule
import torch

from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator


class IMDBDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "../.data/",
        batch_size: int = 64,
        num_workers: int = 4,
        vocab_size: int = 25_000,
        pretrained: str = "glove.6B.100d",
        preprocessing=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = vocab_size
        self.pretrained = pretrained

        self.preprocessing = preprocessing

    def prepare_data(self):
        # Create Fields
        TEXT = Field(tokenize="spacy", include_lengths=True)
        LABEL = LabelField(dtype=torch.float, is_target=True, unk_token=None)

        if self.preprocessing is not None:
            TEXT = Field(
                tokenize="spacy", include_lengths=True, preprocessing=self.preprocessing
            )
        if not (Path("LABEL.pt").exists() and Path("TEXT.pt").exists()):
            # Download
            IMDB.download(root=self.data_dir)

            IMDB_full = IMDB(self.data_dir, text_field=TEXT, label_field=LABEL)
            IMDB_train, IMDB_test = IMDB_full.splits(TEXT, LABEL)

            # Build vocab
            print("Building Vocabulary...")
            TEXT.build_vocab(
                IMDB_train,
                max_size=self.vocab_size,
                vectors=self.pretrained,
                unk_init=torch.Tensor.normal_,
            )

            torch.save(TEXT.vocab, Path(self.data_dir) / "TEXT.pt")

            LABEL.build_vocab(IMDB_train)
            torch.save(LABEL.vocab, Path(self.data_dir) / "LABEL.pt")

    def setup(self, stage=None):
        self.TEXT = Field(tokenize="spacy", include_lengths=True)
        self.LABEL = LabelField(dtype=torch.float, is_target=True, unk_token=None)
        self.TEXT.vocab = torch.load(Path(self.data_dir) / "TEXT.pt")
        self.LABEL.vocab = torch.load(Path(self.data_dir) / "LABEL.pt")

        IMDB_full = IMDB(self.data_dir, text_field=self.TEXT, label_field=self.LABEL)
        self.IMDB_train, self.IMDB_test = IMDB_full.splits(self.TEXT, self.LABEL)
        self.IMDB_train, self.IMDB_val = self.IMDB_train.split(split_ratio=[0.8, 0.2])

    def train_dataloader(self):
        return BucketIterator(
            self.IMDB_train, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return BucketIterator(
            self.IMDB_test, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return BucketIterator(
            self.IMDB_test, batch_size=self.batch_size, shuffle=False
        )