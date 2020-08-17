import argparse
import pickle

import torch

from models import *
import spacy

from utils import count_parameters

nlp = spacy.load('en')

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='fast', type=str, help='pre-trained model architecture')

args = parser.parse_args()


def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [text_vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    packed_info = (tensor, length_tensor)
    prediction = torch.sigmoid(model(packed_info))
    pred = prediction.item()
    print('Score: ',pred)
    xd = int(pred > 0.5)
    return label_vocab.itos[xd]


if __name__ == '__main__':
    model_arch = args.model

    pretrained_folder = "pretrained/" + model_arch + "/"
    ckpt = pretrained_folder + "checkpoints/trained.ckpt"
    if model_arch == 'simple':
        model = SimpleRNNClassifier.load_from_checkpoint(ckpt, map_location=torch.device('cpu'))
    elif model_arch == 'birnn':
        model = BiLSTMClassifier.load_from_checkpoint(ckpt, map_location=torch.device('cpu'))
    elif model_arch == 'fast':
        model = FastClassifier.load_from_checkpoint(ckpt, map_location=torch.device('cpu'))
    else:
        raise ValueError("The model doesn't exist, use | simple | birnn | fast |")
        exit(1)

    print('Parameters')
    print('----------')
    print(count_parameters(model))
    print('----------')

    print('Loading Vocabulary...')
    with open(pretrained_folder + 'text.pkl', 'rb') as f:
        text_vocab = pickle.load(f)
    with open(pretrained_folder + 'label.pkl', 'rb') as f:
        label_vocab = pickle.load(f)

    while True:
        text = input('Enter Text..\n')
        print(predict_sentiment(text))
