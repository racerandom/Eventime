# coding=utf-8
from typing import Dict, Tuple, Sequence, List, Generic

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import TempUtils
from TempData import Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lstm2vec(nn.Module):

    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 bidirectional: bool = True,
                 num_layer: int = 1,
                 out_droprate: float = 0.3) -> None:

        nn.Module.__init__(self)

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layer = num_layer
        self.lstm_model = nn.LSTM(input_dim,
                                  hidden_dim // 2,
                                  num_layers=num_layer,
                                  batch_first=True,
                                  bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=out_droprate)

    @staticmethod
    def init_rnn_hidden(batch_size: int,
                        hidden_dim: int,
                        num_layer: int,
                        bidirectional: bool) -> Tuple[torch.tensor, torch.tensor]:
        bi_num = 2 if bidirectional else 1
        return (torch.zeros(num_layer * bi_num, batch_size, hidden_dim // bi_num).to(device),
                torch.zeros(num_layer * bi_num, batch_size, hidden_dim // bi_num).to(device))

    def forward(self, embed_input) -> torch.tensor:

        batch_size = embed_input.shape[0]
        rnn_input = embed_input
        rnn_hidden = self.init_rnn_hidden(batch_size, self.hidden_dim, self.num_layer, self.bidirectional)
        rnn_out, _ = self.lstm_model(rnn_input, rnn_hidden)
        rnn_out = self.dropout(F.relu(rnn_out))
        return rnn_out


class EntityAttn(nn.Module):

    def __init__(self, attn_input_dim: int,
                 attn_fc_dim: int,
                 attn_fc_droprate: float,
                 attn_out_droprate: float) -> None:
        nn.Module.__init__(self)
        self.attn_fc1 = nn.Linear(attn_input_dim, attn_fc_dim)
        self.attn_fc1_drop = nn.Dropout(p=attn_fc_droprate)
        self.attn_fc2 = nn.Linear(attn_fc_dim, 1)
        self.attn_out_drop = nn.Dropout(p=attn_out_droprate)

    def forward(self, model_out: torch.tensor, event_rep: torch.tensor, timex_rep: torch.tensor) -> torch.tensor:

        max_seq_len = model_out.shape[1]

        attn_in_list = [model_out]
        if event_rep:
            attn_in_list.append(event_rep.repeat(1, max_seq_len, 1))
        elif timex_rep:
            attn_in_list.append(timex_rep.repeat(1, max_seq_len, 1))

        attn_in = torch.cat(attn_in_list, dim=-1)
        attn_fc1_out = self.attn_fc1_drop(F.relu(self.attn_fc1(attn_in)))
        align_W = F.softmax(self.attn_fc2(attn_fc1_out), dim=1).transpose(1, 2)
        attn_out = self.attn_out_drop(torch.bmm(align_W, model_out).squeeze(1))
        return attn_out


class OutLayer(nn.Module):

    def __init__(self,
                 model_out_dim: int,
                 fc1_hidden_dim: int,
                 fc1_droprate: float,
                 vocab: Vocabulary) -> None:

        nn.Module.__init__(self)

        self.vocab = vocab

        self.fc1_layer = nn.Linear(model_out_dim, fc1_hidden_dim)

        self.fc1_dropout = nn.Dropout(p=fc1_droprate)

        self.fc2_layer = nn.Linear(fc1_hidden_dim, len(vocab._token_to_index['labels']) * 4)

    def forward(self,
                model_out: torch.tensor) -> torch.tensor:

        fc1_out = F.relu(self.fc1_layer(model_out))

        fc1_out = self.fc1_dropout(fc1_out)

        fc2_out = self.fc2_layer(fc1_out)

        if len(self.vocab.get_token_to_index()['labels']) == 4:
            pred_prob = F.log_softmax(fc2_out.view(-1, 4, 4), dim=-1)
        elif len(self.vocab.get_token_to_index()['labels']) == 2:
            pred_prob = F.sigmoid(fc2_out)
        else:
            raise Exception('[ERROR] Unknown update label...')
        return pred_prob


class TempClassifier(nn.Module):

    def __init__(self, embedding_dict: Dict[str, nn.Embedding],
                 model: Lstm2vec,
                 attn_layer: EntityAttn,
                 out_layer: OutLayer) -> None:
        super(TempClassifier, self).__init__()
        for namespace, embed in embedding_dict.items():
            setattr(self, '%s_embedding' % namespace, embed)
        self.model = model
        if attn_layer:
            self.attn_layer = attn_layer
        self.out_layer = out_layer

    def forward(self, field_dic: Dict[str, torch.tensor]) -> torch.tensor:

        input_list = [getattr(self, '%s_embedding' % n)(in_t) for n, in_t in field_dic.items() if 'masks' not in n]
        embed_input = torch.cat(input_list,dim=-1)

        model_out = self.model(embed_input)

        if self.attn_layer:

            # get event/timex mention reps
            event_rep, timex_rep = None, None

            if 'event_masks' in field_dic:
                event_masks = field_dic['event_masks']
                event_rep = torch.bmm(event_masks.unsqueeze(1), embed_input)

            if 'timex_masks' in field_dic:
                timex_masks = field_dic['timex_masks']
                timex_rep = torch.bmm(timex_masks.unsqueeze(1), embed_input)

            aggregate_out = self.attn_layer(model_out, event_rep, timex_rep)
        else:
            aggregate_out = model_out.max(dim=1)[0]

        return self.out_layer(aggregate_out)


class TempRNN(nn.Module):

    def __init__(self, word_size, dist_size, targ2ix,
                 max_sent_len, pre_embed, **params):

        nn.Module.__init__(self)

        self.params = params

        if isinstance(pre_embed, np.ndarray):
            self.word_embeddings, self.word_dim = TempUtils.pre_to_embed(pre_embed, freeze_mode=self.params['freeze_mode'])

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

    @staticmethod
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


class TempAttnRNN(nn.Module):

    def __init__(self, word_size, dist_size, targ2ix,
                 max_sent_len, pre_embed, **params):

        nn.Module.__init__(self)

        self.params = params

        if isinstance(pre_embed, np.ndarray):
            self.word_embeddings, self.word_dim = TempUtils.pre_to_embed(pre_embed, freeze_mode=self.params['freeze_mode'])

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
