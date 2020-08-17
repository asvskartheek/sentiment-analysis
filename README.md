# sentiment-analysis
Sentiment Analysis using Pytorch-Lightning

This is heavily inspired by bentrevett's [work](https://github.com/bentrevett/pytorch-sentiment-analysis). I made this so that it becomes easier for people who are
new to NLP and pytorch-lightning, to get them going with a very basic framework to get started.

With this, you can
1. Train popular architectures to according to your hyperparameters (on IMDb dataset for now, planning to extend)
2. Train your own architecture. (Instructions on how this can be done will be added soon)
3. Use one of the pre-trained model, to just get the setiment prediction of the popular models.

## Pre-Trained Models Available
### 1. Simple RNN Classifier
### 2. Bi-Directinal LSTM Classifier
### 3. Fast Classifier

## How to train pre-trained models
This will give a list of all the hyperparameters you can tweak to check and have fun.
```
python train.py --help
```
### DEFAULTS
**Architecture:** Average Pooling over embeddings and a linear classifier over it. This model was introduced by the paper [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759). Our implementation is almost verbatim from [bentrevett](https://github.com/bentrevett/pytorch-sentiment-analysis)
**Optimizer:** Adam with 1e-3 learning rate.
**DEVICE**: The code by default assumes that you are using CPU. To ue a GPU, you have to run with the flag --n_gpus=1, if you are using a GPU we suggest to use a batch size of 64.
