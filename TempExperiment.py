# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class softmax_layer(nn.Module):

    def __init__(self, model_out_dim, targ_size):
        super(softmax_layer, self).__init__()
        self.fc = nn.Linear(model_out_dim, targ_size)

    def forward(self, model_out):
        fc_out = self.fc(model_out)
        softmax_out = F.log_softmax(fc_out, dim=1)
        return softmax_out

class baseRNN(baseConfig, nn.Module):

    def __init__(self, word_size, dsdp_size, targ_size,
                 max_sent_len, max_sdp_len, pre_embed, **params):

        baseConfig.__init__(
            self, word_size, targ_size,
            max_sent_len, **params
        )
        nn.Module.__init__(self)

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = REData.pre2embed(pre_embed, freeze_mode=self.params['freeze_mode'])

        self.input_dropout = nn.Dropout(p=self.params['input_dropout'])

        self.rnn_input_dim = self.word_dim

        self.rnn = nn.LSTM(self.rnn_input_dim,
                           self.rnn_hidden_dim // 2,
                           num_layers=self.params['rnn_layer'],
                           batch_first=True,
                           bidirectional=True)

        self.rnn_dropout = nn.Dropout(p=self.params['rnn_dropout'])

        if self.params['ranking_loss']:
            self.output_layer = ranking_layer(self.rnn_hidden_dim,
                                              targ_size)
        else:
            self.output_layer = softmax_layer(self.rnn_hidden_dim,
                                              targ_size)

    def forward(self, *tensor_feats):

        batch_size = tensor_feats[0].shape[0]

        word_embed_input = self.word_embeddings(tensor_feats[0])

        rnn_input = self.input_dropout(word_embed_input)

        rnn_hidden = self.init_rnn_hidden(batch_size,
                                          self.rnn_hidden_dim,
                                          num_layer=self.params['rnn_layer'])

        rnn_out, rnn_hidden = self.rnn(rnn_input, rnn_hidden)

        # fc_in = torch.cat(torch.unbind(rnn_hidden[0],dim=0), dim=1) ## last hidden state

        rnn_out = self.rnn_dropout(catOverTime(rnn_out, 'max'))

        model_out = self.output_layer(rnn_out)

        return model_out