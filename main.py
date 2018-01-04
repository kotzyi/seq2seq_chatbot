# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import load
import models
import config as cfg
import train
import communicate as comm

corpus_QA, pairs = load.prepareData('ep')
load.updateData(corpus_QA,'dict')

"""
encoder1 = models.EncoderRNN(corpus_QA.n_words, cfg.hidden_size)
#attn_decoder1 = models.AttnDecoderRNN(cfg.hidden_size, corpus_QA.n_words,1, dropout_p=cfg.dropout_p)
attn_decoder1 = models.LuongAttnDecoderRNN('general',cfg.hidden_size, corpus_QA.n_words, 2)

if cfg.use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

train.trainIters(encoder1, attn_decoder1, corpus_QA, pairs, n_iters=2000, print_every=200)

while True:
    s = input("Q: ")
    s = s.strip()
    pairs = [[s,' ']]
    print(pairs)
    train.evaluateRandomly(encoder1, attn_decoder1, corpus_QA, pairs, n=1)
"""
# Initialize models
encoder = models.EncoderRNN(corpus_QA.n_words, cfg.hidden_size, cfg.n_layers, dropout=cfg.dropout)
decoder = models.LuongAttnDecoderRNN(cfg.attn_model, cfg.hidden_size, corpus_QA.n_words, cfg.n_layers, dropout=cfg.dropout)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=cfg.learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=cfg.learning_rate * cfg.decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

# Move models to GPU
if cfg.USE_CUDA:
    encoder.cuda()
    decoder.cuda()

import sconce
job = sconce.Job('seq2seq-translate', {
    'attn_model': cfg.attn_model,
    'n_layers': cfg.n_layers,
    'dropout': cfg.dropout,
    'hidden_size': cfg.hidden_size,
    'learning_rate': cfg.learning_rate,
    'clip': cfg.clip,
    'teacher_forcing_ratio': cfg.teacher_forcing_ratio,
    'decoder_learning_ratio': cfg.decoder_learning_ratio,
})
job.plot_every = cfg.plot_every
job.log_every = cfg.print_every


# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

# Begin!
ecs = []
dcs = []
eca = 0
dca = 0
epoch = 1

while epoch < cfg.n_epochs:
    epoch += 1
    
    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = train.random_batch(corpus_QA, pairs, cfg.batch_size)

    # Run the train function
    loss, ec, dc = train.train(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion, batch_size = cfg.batch_size,
    )

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss
    eca += ec
    dca += dc
    
    job.record(epoch, loss)

    if epoch % cfg.print_every == 0:
        print_loss_avg = print_loss_total / cfg.print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (cfg.time_since(start, epoch / cfg.n_epochs), epoch, epoch / cfg.n_epochs * 100, print_loss_avg)
        print(print_summary)


addr = ('0.0.0.0',0)
connection = comm.Communicate(cfg.TCP_IP,cfg.TCP_PORT)
connection.listen()	

while True:
    s, addr = connection.receive(addr)
    s = s.decode()
    s = s.strip()
    print(s)
    pairs = [s,' ']
    s = train.evaluateRandomly(encoder, decoder, corpus_QA, pairs)
    print(s)
    connection.send(addr, s.encode())


"""
while True:
    s = input("Q: ")
    s = s.strip()
    pairs = [s,' ']
    train.evaluateRandomly(encoder, decoder, corpus_QA, pairs)
"""
