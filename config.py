import torch
import time
import math

TCP_IP = '121.134.144.52'#'10.214.35.36'
TCP_PORT = 5005
BUFFER_SIZE = 4096

PAD_token = 0
SOS_token = 1
EOS_token = 2

MAX_LENGTH = 50

USE_CUDA = torch.cuda.is_available()

# Configure models
attn_model = 'dot'
hidden_size = 500
n_layers = 3
dropout = 0.1
batch_size = 40

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.00025
decoder_learning_ratio = 5.0
n_epochs = 3000
epoch = 0
plot_every = 20
print_every = 100
evaluate_every = 1000

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
