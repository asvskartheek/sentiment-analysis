from pathlib import Path

from pytorch_lightning import LightningDataModule
import torch
from torchtext.data import Field, LabelField, BucketIterator
from torchtext.datasets import SST


class SSTDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../.data/",
            batch_size: int = 64,
            num_workers: int = 4,
            vocab_size: int = 25_000,
            pretrained: str = "glove.6B.100d"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = vocab_size
        self.pretrained = pretrained

    def prepare_data(self):
        if not (Path('sst_LABEL.pt').exists() and Path('sst_TEXT.pt').exists()):
            # Download
            SST.download(root=self.data_dir)

            # Build vocab
            print("Building Vocabulary...")
            TEXT = Field(tokenize="spacy", include_lengths=True)
            LABEL = LabelField(dtype=torch.float, is_target=True, unk_token=None)

            SST_train, SST_valid, SST_test = SST.splits(TEXT, LABEL)

            TEXT.build_vocab(
                SST_train,
                max_size=self.vocab_size,
                vectors=self.pretrained,
                unk_init=torch.Tensor.normal_,
            )

            torch.save(TEXT.vocab, Path(self.data_dir) / 'sst_TEXT.pt')

            LABEL.build_vocab(SST_train)
            torch.save(LABEL.vocab, Path(self.data_dir) / 'sst_LABEL.pt')

    def setup(self, stage=None):
        self.TEXT = Field(tokenize="spacy", include_lengths=True)
        self.LABEL = LabelField(dtype=torch.float, is_target=True, unk_token=None)
        self.TEXT.vocab = torch.load(Path(self.data_dir) / 'TEXT.pt')
        self.LABEL.vocab = torch.load(Path(self.data_dir) / 'LABEL.pt')

        # SST_full = SST(Path(self.data_dir) / 'trees/', text_field=self.TEXT, label_field=self.LABEL)
        self.SST_train, self.SST_valid, self.SST_test = SST.splits(self.TEXT, self.LABEL)

    def train_dataloader(self):
        return BucketIterator(
            self.SST_train, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return BucketIterator(
            self.SST_valid, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return BucketIterator(
            self.SST_test, batch_size=self.batch_size, shuffle=False
        )