import itertools
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from matplotlib import pyplot as plt
import pandas as pd
from torch.autograd import Variable

def run_transformer(device, lr, epochs, dropout, N_sen, N_asp, h_sen, h_asp,
                    d_ff1, d_ff2, train_iterator, test_iterator, vocab):
    d_model = 300

    model = make_model(vocab, N_sen, N_asp, d_model, d_ff1, d_ff2,
                       h_sen, h_asp, dropout, device).to(device)
    loss_func = nn.NLLLoss()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr)
    return fit(epochs, model, loss_func, opt, train_iterator, test_iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def categorical_correct(preds, y):
    """
    Returns number of correct predictions per batch.
    """
    # get the index of the max probability:
    max_preds = preds.argmax(dim = 1, keepdim = True)
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum()

def loss_batch(model, loss_func, sen, asp, asp_position, polarity, asp_pos_itos, opt=None):
    asp_position = torch.tensor([eval(asp_pos_itos[position_i]) for position_i in asp_position])

    pred = model(sen, asp, asp_position)
    loss = loss_func(pred, polarity)
    correct = categorical_correct(pred, polarity)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), correct, len(sen), pred, polarity

def pred_real_table(preds, y, polarity_stoi):
    table = pd.DataFrame(columns=['positive', 'neutral', 'negative'],
                         index=['positive', 'neutral', 'negative'])
    preds = pd.Series(torch.cat(preds).cpu().argmax(
            dim=1, keepdim=True).squeeze(1), name='preds')
    reals = pd.Series(torch.cat(y).cpu(), name='reals')
    df = pd.concat([preds, reals], axis=1)
    n = len(df)
    for pred, real in itertools.product(table.index, table.columns):
        table.at[pred, real] = len(df[(df['preds']==polarity_stoi[pred]) &
                (df['reals']==polarity_stoi[real])])
    table['total predicted'] = table.sum(axis=1).astype(int)
    table.loc['total realizations']= table.sum().astype(int)
    table = table.applymap(lambda x: str(x)+' ('+'{:3.0f}'.format(float(x)/n*100)+'%)')
    return table

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    all_train_losses = []
    all_val_losses = []
    asp_pos_itos = train_dl.dl.dataset.fields['AspectPosition'].vocab.itos
    polarity_stoi = train_dl.dl.dataset.fields['Polarity'].vocab.stoi

    print('Training model for {:d} epochs...'.format(epochs))
    start_time = time.time()
    start_epoch_time = start_time

    for epoch in range(epochs):
        model.train()
        train_losses, train_correct, train_nums, train_preds, train_y = zip(
                *[loss_batch(model, loss_func, sen, asp, asp_position, polarity, asp_pos_itos, opt) for sen, asp, asp_position, polarity in train_dl]
            )
        epoch_train_correct = np.sum(train_correct, dtype=float)
        epoch_train_accuracy = epoch_train_correct / np.sum(train_nums)
        epoch_train_loss =  np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
        all_train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_losses, val_correct, val_nums, val_preds, val_y = zip(
                *[loss_batch(model, loss_func, sen, asp, asp_position, polarity, asp_pos_itos) for sen, asp, asp_position, polarity in valid_dl]
            )

        epoch_val_correct = np.sum(val_correct, dtype=float)
        epoch_val_accuracy = epoch_val_correct / np.sum(val_nums)
        epoch_val_loss = np.sum(np.multiply(val_losses, val_nums)) / np.sum(val_nums)
        all_val_losses.append(epoch_val_loss)

        SHOW_FREQ = 1 # Determines how often to print epoch results
        if (epoch+1) % SHOW_FREQ == 0:
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_epoch_time, end_time)
            total_mins, total_secs = epoch_time(start_time, end_time)

            print(f'\n\nEpoch: {epoch+1} of {epochs}'
                  f' | Time since last: {epoch_mins}m {epoch_secs}s'
                  f' | Total ML time: {total_mins}m {total_secs}s')
            print(f'\tML Train Loss: {epoch_train_loss:.3f}'
                  f' | ML Train Acc: {epoch_train_accuracy*100:.2f}%')
            print(f'\t ML Val Loss: {epoch_val_loss:.3f}'
                  f' |  ML Val Acc: {epoch_val_accuracy*100:.2f}%')
            print('\nTrain prediction-realization table:')
            print(pred_real_table(train_preds, train_y, polarity_stoi))
            print('\nVal prediction-realization table:')
            print(pred_real_table(val_preds, val_y, polarity_stoi))
            start_epoch_time = time.time()

    # Plot validation loss over epochs
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title("Loss vs. Epochs")
    plt.tight_layout()
    plt.plot(all_train_losses)
    plt.plot(all_val_losses)
    plt.legend(['Train loss', 'Test loss'], loc='upper right')
    plt.show()

    return model, epoch_train_loss, epoch_val_loss, epoch_train_accuracy, \
            epoch_val_accuracy, epoch_train_correct, epoch_val_correct

def make_model(vocab, N_sen, N_asp, d_model, d_ff1, d_ff2, h_sen,
               h_asp, dropout, device):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn_sen = MultiHeadedAttention(h_sen, d_model, dropout)
    attn_asp = MultiHeadedAttention(h_asp, d_model, dropout)
    ff1 = PositionwiseFeedForward(d_model, d_ff1, dropout)
    ff2 = PositionwiseFeedForward(d_model, d_ff2, dropout)
    enc_layer_sen = EncoderLayer(d_model, attn_sen, c(ff1), dropout)
    enc_layer_asp = EncoderLayer(d_model, attn_asp, c(ff1), dropout)
    position = PositionalEncoding(d_model, device)
    pad_idx = vocab.stoi['<pad>']
    unk_idx = vocab.stoi['<unk>']
    model = FullModel(
            Embeddings(d_model, vocab, pad_idx, unk_idx),
            position,
            InputEncoder(
                    Encoder(enc_layer_sen, N_sen),
                    Encoder(enc_layer_asp, N_asp)),
            c(ff2),
            AspectSpecificSentence(d_model, dropout, device),
            Generator(d_model))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class FullModel(nn.Module):
    "Combine all the different modules in one model."
    def __init__(self, embedder, position_encoder, input_encoder, fnn2,
                 asp_spec_sen, generator):
        super(FullModel, self).__init__()
        self.embed = embedder
        self.position = position_encoder
        self.input_encoder = input_encoder
        self.fnn2 = fnn2
        self.asp_spec_sen = asp_spec_sen
        self.generator = generator

    def forward(self, sen, asp, asp_position):
        sen_encod, asp_encod = self.encode_input(sen, asp, asp_position)
        sen_att = self.att_calc(sen_encod, asp_encod)
        sen_fnn2 = self.fnn2(sen_att)
        return self.generator(sen_fnn2)

    def encode_input(self, sen, asp, asp_position):
        sen_embed = self.embed(sen)
        return self.input_encoder(*self.position(sen_embed, asp_position))

    def att_calc(self, sen_encod, asp_encod):
        return self.asp_spec_sen(sen_encod, asp_encod)

class Generator(nn.Module):
    "Define standard linear + log-softmax generation step."
    def __init__(self, d_model):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, 3)
        self.logsoftmax = nn.LogSoftmax(dim = -1)

    def forward(self, x):
        final_lin = self.proj(x)
        return self.logsoftmax(final_lin)

class AspectSpecificSentence(nn.Module):
    "Create the aspect-specific representation of the sentence."
    def __init__(self, d_model, dropout, device):
        super(AspectSpecificSentence, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.norm = LayerNorm(d_model)
        self.attention = MultiHeadedAttention(1, d_model, dropout)
        self.sublayer = SublayerConnection(d_model, dropout)

    def forward(self, sen_encod, asp_encod):
        sen_att = self.sublayer(sen_encod, lambda sen_encod: self.attention(sen_encod, asp_encod, asp_encod))
        return torch.mean(sen_att, dim=1)

class InputEncoder(nn.Module):
    "Encode the input sentences and input aspects in two different encoders."
    def __init__(self, sen_encoder, asp_encoder):
        super(InputEncoder, self).__init__()
        self.sen_encoder = sen_encoder
        self.asp_encoder = asp_encoder

    def forward(self, sen_embed, asp_embed):
        "Take in and process input sentences and aspects"
        sen_encod = self.sen_encoder(sen_embed)
        asp_encod = self.asp_encoder(asp_embed)
        return sen_encod, asp_encod

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        "Pass the input through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "Let the data flow through the encoder sub-layers"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements self-attention for h heads"
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.tanh(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, pad_idx, unk_idx):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = len(vocab)
        self.emb = nn.Embedding(self.vocab_size, d_model, padding_idx = pad_idx)
        self.emb.weight.data.copy_(vocab.vectors) # load pretrained vectors
        self.emb.weight.data[pad_idx] = torch.zeros(d_model)
        self.emb.weight.requires_grad = False # make embedding non trainable

    def forward(self, sen):
        sen = sen.transpose(0,1)
        out = self.emb(sen)
        out = out.transpose(0,1)
        return out * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.device = device

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, sen, asp_position):
        sen_embed = sen + Variable(self.pe[:, :sen.size(1)], requires_grad=False)

        (asp_from, asp_to) = asp_position[:,0], asp_position[:,1]
        max_asp_len = torch.max(asp_to-asp_from)
        asp_embed = torch.ones(len(sen), max_asp_len, self.d_model).to(self.device)
        for example in range(len(sen)):
            for i, word in enumerate(range(asp_from[example], asp_to[example])):
                asp_embed[example,i,:] = sen_embed[example, word, :]

        return sen_embed, asp_embed