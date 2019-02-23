# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import TempUtils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TempAttnRNN(nn.Module):

    def __init__(self, word_size, dist_size, targ2ix,
                 max_sent_len, pre_embed, **params):

        nn.Module.__init__(self)

        self.params = params

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = TempUtils.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.dist_size = dist_size

        if self.dist_size > 0:
            self.dist_embeddings = nn.Embedding(dist_size, self.params['dist_dim'])
            self.rnn_input_dim = self.word_dim + self.params['dist_dim']
        else:
            self.rnn_input_dim = self.word_dim

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.params['rnn_hidden_dim'] // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.attn_W = torch.nn.Parameter(torch.empty(self.params['rnn_hidden_dim']).uniform_(-0.1, 0.1))

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc1_layer = nn.Linear(self.params['rnn_hidden_dim'], self.params['fc1_hidden_dim'])

        self.fc1_dropout = nn.Dropout(p=self.params['fc1_dropout'])

        self.fc2_layer = nn.Linear(self.params['fc1_hidden_dim'], len(targ2ix) * 4)

    def init_rnn_hidden(self, batch_size, hidden_dim, num_layer=1, bidirectional=True):
        bi_num = 2 if bidirectional else 1
        return (torch.zeros(num_layer * bi_num, batch_size, hidden_dim // bi_num).to(device),
                torch.zeros(num_layer * bi_num, batch_size, hidden_dim // bi_num).to(device))

    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        rnn_input = self.word_embeddings(tensor_feats[0])

        if self.dist_size > 0:
            for t_f in tensor_feats[1:3]:
                rnn_input = torch.cat((rnn_input, self.dist_embeddings(t_f)), dim=-1)
            rnn_input = self.input_dropout(rnn_input)
        else:
            rnn_input = self.input_dropout(rnn_input)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.params['rnn_hidden_dim'],
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)

        # rnn_out = self.rnn_dropout(catOverTime(F.relu(rnn_out), 'max'))

        # rnn_out = rnn_out[:, :, :self.params['rnn_hidden_dim'] // 2] + rnn_out[:, :, self.params['rnn_hidden_dim'] // 2:]

        rnn_out = self.rnn_dropout(F.relu(rnn_out))

        # attn_bW = self.attn_W.repeat(batch_size, 1)
        # attn_alpha = torch.bmm(rnn_out, attn_bW.unsqueeze(2)).squeeze(2)
        attn_alpha = torch.einsum('bsd,d->bs', (rnn_out, self.attn_W))
        attn_prob = F.softmax(attn_alpha, dim=1)
        attn_out = F.tanh(torch.bmm(attn_prob.unsqueeze(1), rnn_out)).squeeze(1)

        fc1_out = F.relu(self.fc1_layer(attn_out))

        fc1_out = self.fc1_dropout(fc1_out)

        fc2_out = self.fc2_layer(fc1_out)

        model_out = F.log_softmax(fc2_out.view(-1, 4, 4), dim=-1)
        # print(model_out.shape)

        return model_out


class TempRNN(nn.Module):

    def __init__(self, word_size, dist_size, targ2ix,
                 max_sent_len, pre_embed, **params):

        nn.Module.__init__(self)

        self.params = params

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = TempUtils.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.dist_size = dist_size
        if self.dist_size > 0:
            self.dist_embeddings = nn.Embedding(dist_size, self.params['dist_dim'])
            self.rnn_input_dim = self.word_dim + self.params['dist_dim']
        else:
            self.rnn_input_dim = self.word_dim

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.params['rnn_hidden_dim'] // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        self.fc1_layer = nn.Linear(self.params['rnn_hidden_dim'], self.params['fc1_hidden_dim'])

        self.fc1_dropout = nn.Dropout(p=self.params['fc1_dropout'])

        self.fc2_layer = nn.Linear(self.params['fc1_hidden_dim'], len(targ2ix) * 4)

    def init_rnn_hidden(self, batch_size, hidden_dim, num_layer=1, bidirectional=True):
        bi_num = 2 if bidirectional else 1
        return (torch.zeros(num_layer * bi_num, batch_size, hidden_dim // bi_num).to(device),
                torch.zeros(num_layer * bi_num, batch_size, hidden_dim // bi_num).to(device))

    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        rnn_input = self.word_embeddings(tensor_feats[0])

        if self.dist_size > 0:
            for t_f in tensor_feats[1:3]:
                rnn_input = torch.cat((rnn_input, self.dist_embeddings(t_f)), dim=-1)
            rnn_input = self.input_dropout(rnn_input)
        else:
            rnn_input = self.input_dropout(rnn_input)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.params['rnn_hidden_dim'],
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)

        rnn_out = self.rnn_dropout(F.relu(rnn_out).max(dim=1)[0])

        fc1_out = F.relu(self.fc1_layer(rnn_out))

        fc1_out = self.fc1_dropout(fc1_out)

        fc2_out = self.fc2_layer(fc1_out)

        # print(fc2_out.shape)
        if self.params['update_label'] == 3:
            pred_prob = F.log_softmax(fc2_out.view(-1, 4, 4), dim=-1)
        elif self.params['update_label'] == 1:
            m = torch.nn.Sigmoid()
            pred_prob = m(fc2_out)
        else:
            raise Exception('[ERROR] Unknown update label...')
        return pred_prob
