from birnn import BiLSTMClassifier
import pytorch_lightning as pl
import torch

# import torch
# from torchtext import data
#
# EMBED_DIM = 100
# NUM_LAYERS = 2
# HIDDEN_DIM = 256
# OUTPUT_DIM = 1
# PADDING_IDX = 1
# DROPOUT_RATE = 0.5
#
# TEXT = data.Field(lower=True, include_lengths=True)
# LABEL = data.Field(sequential=False, dtype=torch.float, is_target=True, unk_token=None)
#
#
# def predict_sentiment(model, sentence):
#     model.eval()
#     x = TEXT.tokenize(sentence)
#     tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
#     indexed = [TEXT.vocab.stoi[t] for t in tokenized]
#     length = [len(indexed)]
#     tensor = torch.LongTensor(indexed)
#     tensor = tensor.unsqueeze(1)
#     length_tensor = torch.LongTensor(length)
#     prediction = torch.sigmoid(model(tensor, length_tensor))
#     return prediction.item()
from fast import FastClassifier

model = FastClassifier.load_from_checkpoint(
    checkpoint_path='./lightning_logs/version_5/checkpoints/epoch=4.ckpt',
    hparams_file='./lightning_logs/version_5/checkpoints/hparams.yaml',
    map_location=torch.device('cpu')
)

# model = BiLSTMClassifier()
# trainer = pl.Trainer(resume_from_checkpoint='lightning_logs/bi (epoch=5)/checkpoints/epoch=4.ckpt')
# checkpoint = torch.load('lightning_logs/bi (epochs=5)/checkpoints/epoch=4.ckpt', map_location=torch.device('cpu'))
# print(checkpoint['hparams'])
# automatically restores model, epoch, step, LR schedulers, apex, etc...
# trainer.fit(model)