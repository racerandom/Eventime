import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
import numpy as np
import random
import pdb
import gensim

import TempUtils
from TempData import MultipleDatasets

torch.manual_seed(23214)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import random
EPS_THRES = 0

update_strategies = torch.tensor([[0, 0, 0],
                                 [1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1],
                                 [1, 1, 0],
                                 [0, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=torch.long, device=device)


def get_index_of_max(scores):
    index = scores.max(0)[1].item()
    return index

ACTION_TO_IX = {'COL0':0, 'COL1':1, 'COL2':2, 'COL01':3, 'COL12':4, 'COL012':5, 'NONE':6}
IX_TO_ACTION = {v: k for k, v in ACTION_TO_IX.items()}


def select_action(out_score, IX_TO_ACTION):
    sample = random.random()
    if sample > EPS_THRES:
        index = get_index_of_max(out_score)
        return IX_TO_ACTION[index]
    else:
        return IX_TO_ACTION[random.randrange(7)]


def action2out(action, curr_out, curr_time, IX_TO_ACTION):

    def update_col0(curr_out, curr_time):
        curr_out[0] = curr_time
        return curr_out

    def update_col1(curr_out, curr_time):
        curr_out[1] = curr_time
        return curr_out

    def update_col2(curr_out, curr_time):
        curr_out[2] = curr_time
        return curr_out

    def update_col01(curr_out, curr_time):
        curr_out[0] = curr_time
        curr_out[1] = curr_time
        return curr_out

    def update_col12(curr_out, curr_time):
        curr_out[1] = curr_time
        curr_out[2] = curr_time
        return curr_out

    def update_col012(curr_out, curr_time):
        curr_out[0] = curr_time
        curr_out[1] = curr_time
        curr_out[2] = curr_time
        return curr_out

    def update_none(curr_out, curr_time):
        return curr_out

    if action == 'COL0':
        return update_col0(curr_out, curr_time)
    elif action == 'COL1':
        return update_col1(curr_out, curr_time)
    elif action == 'COL2':
        return update_col2(curr_out, curr_time)
    elif action == 'COL01':
        return update_col01(curr_out, curr_time)
    elif action == 'COL12':
        return update_col12(curr_out, curr_time)
    elif action == 'COL012':
        return update_col012(curr_out, curr_time)
    elif action == 'NONE':
        return update_none(curr_out, curr_time)
    else:
        raise Exception("[ERROR]Wrong action!!!")


def batch_action2out(out_scores, norm_times, IX_TO_ACTION, BATCH_SIZE):
    preds_out = torch.ones(BATCH_SIZE, 3) * -1 ## initial prediction
    for i in range(out_scores.size()[0]):
        for j in range(out_scores.size()[1]):
            action = select_action(out_scores[i][j], IX_TO_ACTION)
#             print(action)
            action2out(action, preds_out[i], 0 if i == 0 else 1, IX_TO_ACTION)
    return preds_out


def is_seq_feat(feat_type):
    return feat_type.split('_')[-1] == 'seq'


def is_tok_feat(feat_type):
    return feat_type.split('_')[-1] == 'tok'


def is_sent_feat(feat_type):
    return feat_type.split('_')[-1] == 'sent'


def is_feat(feat_type):
    if isinstance(feat_type, str):
        feat_toks = feat_type.split('_')
        if len(feat_toks) == 3 and feat_toks[0] in ['sour', 'targ', 'full'] and feat_toks[-1] in ['seq', 'tok', 'sent'] and feat_toks[1] in ['word', 'char', 'pos', 'dep', 'dist', 'index']:
            return True
        else:
            return False
    else:
        return False


def which_branch(feat_type):
    return feat_type.split('_')[0]


def which_feat(feat_type):
    return feat_type.split('_')[-2]


def seq_input_dim(params, word_dim):
    return word_dim + params['char_dim'] + params['pos_dim'] + params['dep_dim']


def getPartOfTensor3D(tensor, index, pad_direct="right"):
    """
    Retrieve SDP seq from RNN output
    :param tensor: RNN output seq (batch_size, max_sent_len, hidden_dim)
    :param index:  SDP index (batch_size, max_SDP_len), element value is conll_id, padded with zeros
    :param pad_direct: defaut 'right' to padding zero tensor
    :return:       (batch_size, max_SDP_len, hidden_dim)
    """
    assert tensor.shape[0] == index.shape[0]
    max_seq_len, dim = index.shape[1], tensor.shape[2]
    T = [t[i[i.nonzero().squeeze(dim=1)]-1] for t, i in zip(tensor, index)]
    if pad_direct == "right":
        padded_T = [torch.cat((t, torch.zeros(max_seq_len - len(t), dim).to(device)), dim=0) for t in T if len(t) < max_seq_len]
    else:
        padded_T = [torch.cat((torch.zeros(max_seq_len - len(t), dim).to(device), t), dim=0) for t in T if len(t) < max_seq_len]
    return torch.stack(padded_T)


class TempCNN(nn.Module):
    def __init__(self, max_seq_len, max_tok_len, max_char_len, action_size, feat_types, verbose_level=0, **params):
        super(TempCNN, self).__init__()
        self.verbose_level = verbose_level
        self.embedding_dropout = nn.Dropout(p=params['dropout_emb'])
        self.max_seq_len = max_seq_len
        self.max_tok_len = max_tok_len
        self.max_char_len = max_char_len

        c1_input_dim = 0
        for feat_type in feat_types:
            if is_seq_feat(feat_type):
                if which_feat(feat_type) == 'word':
                    c1_input_dim += params['word_dim']
                elif which_feat(feat_type) == 'dist':
                    c1_input_dim += params['pos_dim']
                elif params['char_emb'] and which_feat(feat_type) in ['char']:
                    c1_input_dim += params['char_dim']
        self.c1 = nn.Conv1d(c1_input_dim, params['filter_nb'], params['kernel_len'])
        self.kernel_dim = self.max_seq_len - params['kernel_len'] + 1
        self.p1 = nn.MaxPool1d(self.kernel_dim)
        self.conv_dropout = nn.Dropout(p=params['dropout_conv'])

        # self.tok_p1 = nn.MaxPool1d(max_tok_len)
        self.cat_dropout = nn.Dropout(p=params['dropout_cat'])
        fc1_input_dim = params['filter_nb']
        for feat_type in feat_types:
            if not is_seq_feat(feat_type):
                if which_feat(feat_type) == 'word':
                    if params['cat_word_tok']:
                        fc1_input_dim += params['word_dim']
                elif which_feat(feat_type) == 'dist':
                    if params['cat_dist_tok']:
                        fc1_input_dim += params['pos_dim']
        if params['fc_layer']:
            self.fc1 = nn.Linear(fc1_input_dim, params['fc_hidden_dim'])
            self.fc1_drop = nn.Dropout(p=params['dropout_fc'])
            self.fc2 = nn.Linear(params['fc_hidden_dim'], action_size)
        elif not params['fc_layer']:
            self.fc2 = nn.Linear(fc1_input_dim, action_size)

    def forward(self, feat_types, *feat_inputs, **params):

        ## step 1: concat seq feats
        seq_inputs = []
        
        for feat, feat_type in zip(feat_inputs, feat_types):
            if is_seq_feat(feat_type):
                if self.verbose_level == 2:
                    print(feat_type, feat.shape)
                if which_feat(feat_type) in ['word', 'dist']:
                    seq_inputs.append(feat)
                elif params['char_emb'] and which_feat(feat_type) in ['char']:
                    seq_inputs.append(feat)

        # input shape (batch_size, seq_len, input_dim) => (batch_size, input_dim, seq_len)
        seq_inputs = torch.cat(seq_inputs, dim=2).transpose(1, 2)
        embed_inputs = self.embedding_dropout(seq_inputs)

        ## step 2: conv + maxpool
        c1_out = F.relu(self.c1(embed_inputs))

        if self.verbose_level == 2:
            print("c1_output size:", c1_out.shape)

        p1_out = self.p1(c1_out).squeeze(-1)

        if self.verbose_level == 2:
            print("p1_output size:", p1_out.shape)


        ## step 3: concat tok feats
        cat_inputs = [self.conv_dropout(p1_out)]
        for feat, feat_type in zip(feat_inputs, feat_types):
            if is_tok_feat(feat_type):
                if (which_feat(feat_type) == 'word' and params['cat_word_tok']) or (which_feat(feat_type) == 'dist' and params['cat_dist_tok']):
                    if params['mention_cat'] == 'sum':
                        mention_feat = feat.sum(dim=1)
                    elif params['mention_cat'] == 'max':
                        mention_feat = feat.max(dim=1)[0]
                    elif params['mention_cat'] == 'mean':
                        mention_feat = feat.mean(dim=1)
                        # mention_feat = self.tok_p1(feat.transpose(1, 2)).squeeze(-1)
                    cat_inputs.append(self.cat_dropout(mention_feat))
                    if self.verbose_level == 2:
                        print(feat_type, feat.shape, mention_feat.shape)
        cat_out = torch.cat(cat_inputs, dim=-1)
        if self.verbose_level == 2:
            print("cat_output size:", cat_out.shape)


        ## step4: fc layers
        if params['fc_layer']:
            fc1_out = F.relu(self.fc1(cat_out))
            fc1_out = self.fc1_drop(fc1_out)
            fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)
        elif not params['fc_layer']:
            fc2_out = F.log_softmax(self.fc2(cat_out), dim=1)

        if self.verbose_level == 2:
            print()

        return fc2_out


class TempAttnCNN(nn.Module):

    def __init__(self, max_seq_len, max_tok_len, max_char_len, action_size, feat_types, verbose_level=0, **params):
        super(TempAttnCNN, self).__init__()
        self.verbose_level = verbose_level
        self.embedding_dropout = nn.Dropout(p=params['dropout_emb'])

        self.max_seq_len = max_seq_len
        self.max_tok_len = max_tok_len
        self.max_char_len = max_char_len

        c1_input_dim = 0
        for feat_type in feat_types:
            if is_seq_feat(feat_type):
                if which_feat(feat_type) == 'word':
                    c1_input_dim += params['word_dim']
                elif which_feat(feat_type) == 'dist':
                    c1_input_dim += params['pos_dim']
                elif params['char_emb'] and which_feat(feat_type) in ['char']:
                    c1_input_dim += params['char_dim']

        self.c1 = nn.Conv1d(c1_input_dim, params['filter_nb'], params['kernel_len'])
        self.kernel_dim = self.max_seq_len - params['kernel_len'] + 1
        attn_dim = params['filter_nb'] if params['attn_targ'] == 'filter_nb' else self.kernel_dim
        self.attn_W = torch.nn.Parameter(torch.randn(attn_dim, requires_grad=True))
        self.conv_dropout = nn.Dropout(p=params['dropout_conv'])

        #self.tok_p1 = nn.MaxPool1d(max_tok_len)
        self.cat_dropout = nn.Dropout(p=params['dropout_cat'])
        fc1_input_dim = attn_dim
        for feat_type in feat_types:
            if not is_seq_feat(feat_type):
                if which_feat(feat_type) == 'word':
                    if params['cat_word_tok']:
                        fc1_input_dim += params['word_dim']
                elif which_feat(feat_type) == 'dist':
                    if params['cat_dist_tok']:
                        fc1_input_dim += params['pos_dim']
        if params['fc_layer']:
            self.fc1 = nn.Linear(fc1_input_dim, params['fc_hidden_dim'])
            self.fc1_drop = nn.Dropout(p=params['dropout_fc'])
            self.fc2 = nn.Linear(params['fc_hidden_dim'], action_size)
        elif not params['fc_layer']:
            self.fc2 = nn.Linear(fc1_input_dim, action_size)

    def forward(self, feat_types, *feat_inputs, **params):

        ## step 1: concat seq feats
        seq_inputs = []
        for feat, feat_type in zip(feat_inputs, feat_types):
            if is_seq_feat(feat_type):
                if self.verbose_level == 2:
                    print(feat_type, feat.shape)
                if which_feat(feat_type) in ['word', 'dist']:
                    seq_inputs.append(feat)
                elif params['char_emb'] and which_feat(feat_type) in ['char']:
                    seq_inputs.append(feat)
        ## input (batch_size, seq_len, input_dim) => (batch_size, input_dim, seq_len)
        seq_inputs = torch.cat(seq_inputs, dim=2).transpose(1, 2)
        embed_inputs = self.embedding_dropout(seq_inputs)

        ## step2: conv + attn
        batch_size = embed_inputs.shape[0]
        if self.verbose_level == 2:
            print("embed_input size:", embed_inputs.shape)

        c1_out = F.relu(self.c1(embed_inputs))
        if self.verbose_level == 2:
            print("c1_output size:", c1_out.shape)

        # c1_out: [batch, filter_nb, kernel_out]
        c1_out = c1_out if params['attn_targ'] == 'filter_nb' else c1_out.transpose(1, 2)
        attn_M = F.tanh(c1_out)
        W = self.attn_W.unsqueeze(0).expand(batch_size, -1, -1)  # W: [batch_size, 1, filter_nb]
        attn_alpha = F.softmax(torch.bmm(W, attn_M), dim=2)  # rnn1_alpha: [batch_size, 1, kernel_out]
        attn_out = F.tanh(torch.bmm(c1_out, attn_alpha.transpose(1, 2)))  # attn_out: [batch_size, filter_nb, 1]

        if self.verbose_level == 2:
            print("attn_output size:", attn_out.squeeze(-1).shape)

        ## step 3: concat tok feats
        cat_inputs = [self.conv_dropout(attn_out.squeeze(-1))]
        for feat, feat_type in zip(feat_inputs, feat_types):
            if not is_seq_feat(feat_type):
                if (which_feat(feat_type) == 'word' and params['cat_word_tok']) or (which_feat(feat_type) == 'dist' and params['cat_dist_tok']):
                    if params['mention_cat'] == 'sum':
                        mention_feat = feat.sum(dim=1)
                    elif params['mention_cat'] == 'max':
                        mention_feat = feat.max(dim=1)[0]
                    elif params['mention_cat'] == 'mean':
                        mention_feat = feat.mean(dim=1)
                        # mention_feat = self.tok_p1(feat.transpose(1, 2)).squeeze(-1)
                    cat_inputs.append(self.cat_dropout(mention_feat))
                    if self.verbose_level == 2:
                        print(feat_type, feat.shape, '->', mention_feat.shape)
        cat_out = torch.cat(cat_inputs, dim=-1)

        if self.verbose_level == 2:
            print("cat_output size:", cat_out.shape)

        ## step4: fc layers
        if params['fc_layer']:
            fc1_out = F.relu(self.fc1(cat_out))
            fc1_out = self.fc1_drop(fc1_out)
            fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)
        elif not params['fc_layer']:
            fc2_out = F.log_softmax(self.fc2(cat_out), dim=1)

        if self.verbose_level == 2:
            print()

        return fc2_out


class sentSdpRNN(nn.Module):

    def __init__(self, wvocab_size, cvocab_size, pos_size, dep_size, dist_size, action_size,
                 max_sent_len, max_seq_len, max_mention_len, max_word_len,
                 pre_embed=None,
                 verbose=0, **params):

        super(sentSdpRNN, self).__init__()

        """
        initialize parameters
        """
        self.params = params
        self.link_type = self.params['link_type']
        self.hidden_dim = self.params['seq_rnn_dim']
        self.sent_hidden_dim = self.params['sent_rnn_dim']
        self.max_sent_len = max_sent_len
        self.max_seq_len = max_seq_len
        self.max_mention_len = max_mention_len
        self.max_word_len = max_word_len
        self.verbose = verbose

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = TempUtils.pre2embed(pre_embed)

        self.char_dim = params['char_dim']
        if self.char_dim:
            self.char_embeddings = nn.Embedding(cvocab_size, self.char_dim, padding_idx=0)
            self.char_hidden_dim = params['char_dim']
            self.char_rnn = nn.LSTM(self.char_dim, self.char_hidden_dim // 2,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True)

        self.pos_dim = params['pos_dim']
        if self.pos_dim:
            self.pos_embeddings = nn.Embedding(pos_size, self.pos_dim, padding_idx=0)

        self.dep_dim = params['dep_dim']
        if self.dep_dim:
            self.dep_embeddings = nn.Embedding(dep_size, self.dep_dim, padding_idx=0)

        self.dist_dim = params['dist_dim']
        if self.dist_dim:
            self.dist_embeddings = nn.Embedding(dist_size, self.dist_dim, padding_idx=0)

        self.sent_input_dim = self.word_dim + self.char_dim + self.dist_dim + \
                              (self.dist_dim if self.link_type != 'Event-DCT' else 0)

        # print(self.sent_input_dim)

        self.sent_rnn = nn.LSTM(self.sent_input_dim,
                                self.sent_hidden_dim // 2,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

        if self.params['sent_rnn_pool']:
            self.sent_rnn_pool = nn.MaxPool1d(self.max_sent_len)

        self.seq_input_dim = seq_input_dim(self.params, self.word_dim)

        self.sour_rnn = nn.LSTM(self.seq_input_dim,
                                self.hidden_dim // 2,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

        if self.params['link_type'] not in ['Event-DCT']:
            self.targ_rnn = nn.LSTM(self.seq_input_dim,
                                    self.hidden_dim // 2,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True)

        if self.params['seq_rnn_pool']:
            self.seq_rnn_pool = nn.MaxPool1d(self.max_seq_len)

        self.sour_rnn_drop = nn.Dropout(p=self.params['dropout_sour_rnn'])
        if self.params['link_type'] not in ['Event-DCT']:
            self.targ_rnn_drop = nn.Dropout(p=self.params['dropout_targ_rnn'])

        self.feat_drop = nn.Dropout(p=self.params['dropout_feat'])

        self.fc1_input_dim = (self.sent_hidden_dim if self.params['sent_rnn'] else 0) + \
                             (self.hidden_dim if self.params['sdp_rnn'] else 0) + \
                             (self.hidden_dim if self.params['sdp_rnn'] and self.link_type != "Event-DCT" else 0)

        if self.params['link_type'] not in ['Event-DCT']:
            self.fc1_input_dim *= 2

        self.fc1 = nn.Linear(self.fc1_input_dim, self.params['fc_hidden_dim'])
        self.fc1_drop = nn.Dropout(p=self.params['dropout_fc'])
        self.fc2 = nn.Linear(self.params['fc_hidden_dim'], action_size)


    def init_hidden(self, batch_size, hidden_dim):
        return (torch.zeros(2, batch_size, hidden_dim // 2).to(device),
                torch.zeros(2, batch_size, hidden_dim // 2).to(device))

    def forward(self, **input_dict):

        batch_size = input_dict['sour_word_seq'].shape[0]

        """
        full sent rnn 
        """
        sent_input = []
        for feat_type, feat in {k: v for k, v in input_dict.items() if is_sent_feat(k)}.items():
            if self.word_dim and which_feat(feat_type) == 'word':
                embed_feat = self.word_embeddings(feat)
                sent_input.append(embed_feat)
            elif self.dist_dim and which_feat(feat_type) == 'dist':
                embed_feat = self.dist_embeddings(feat)
                sent_input.append(embed_feat)
            elif self.char_dim and which_feat(feat_type) == 'char':
                embed_char = self.char_embeddings(feat.view(batch_size * self.max_sent_len, self.max_word_len))
                self.char_hidden = self.init_hidden(batch_size * self.max_sent_len, self.char_hidden_dim)
                char_outs, self.char_hidden = self.char_rnn(embed_char, self.char_hidden)
                embed_feat = char_outs[:, -1, :].view((batch_size, self.max_seq_len, -1))
                sent_input.append(embed_feat)
        sent_tensor = torch.cat(sent_input, dim=2)

        self.sent_rnn_hidden = self.init_hidden(batch_size, self.sent_hidden_dim)
        sent_rnn_out, self.sent_rnn_hidden = self.sent_rnn(sent_tensor, self.sent_rnn_hidden)

        # print(sent_rnn_out.shape)

        for feat_type, feat in {k: v for k, v in input_dict.items() if is_seq_feat(k)}.items():
            if which_branch(feat_type) == 'sour' and which_feat(feat_type) == 'index':
                sour_sdp_input = getPartOfTensor3D(sent_rnn_out, feat)
            elif which_branch(feat_type) == 'targ' and which_feat(feat_type) == 'index':
                targ_sdp_input = getPartOfTensor3D(sent_rnn_out, feat)

        print("input tensor of SDP rnn", sour_sdp_input.shape)

        """
        SDP layer
        """
        self.sour_rnn_hidden = self.init_hidden(batch_size, self.hidden_dim)
        sour_sdp_out, self.sour_rnn_hidden = self.sour_rnn(sour_sdp_input, self.sour_rnn_hidden)
        if self.params['seq_rnn_pool']:
            sour_sdp_out = self.seq_rnn_pool(sour_sdp_out.transpose(1, 2)).squeeze()
        else:
            sour_sdp_out = sour_sdp_out[:, -1, :]

        if self.link_type in ['Event-Timex', 'Event-Event']:
            self.targ_rnn_hidden = self.init_hidden(batch_size, self.hidden_dim)
            targ_sdp_out, self.targ_rnn_hidden = self.targ_rnn(targ_sdp_input, self.targ_rnn_hidden)
            if self.params['seq_rnn_pool']:
                targ_sdp_out = self.seq_rnn_pool(targ_sdp_out.transpose(1, 2)).squeeze()
            else:
                targ_sdp_out = targ_sdp_out[:, -1, :]

        # print("output tensor of SDP rnn", sour_sdp_out.shape)

        """
        concatenate seq rnn + sent rnn + feat
        """
        cat_input = [self.feat_drop(sent_rnn_out[:, -1, :])]

        cat_input.append(self.sour_rnn_drop(sour_sdp_out))
        if self.link_type in ['Event-Timex', 'Event-Event']:
            cat_input.append(self.targ_rnn_drop(targ_sdp_out))

        # feat_input = []
        # for feat_type, feat in {k: v for k, v in input_dict.items() if is_feat(k) and is_tok_feat(k)}.items():
        #
        #     if self.word_dim and which_feat(feat_type) == 'word':
        #         embed_feat = self.word_embeddings(feat)
        #     elif self.pos_dim and which_feat(feat_type) == 'pos':
        #         embed_feat = self.pos_embeddings(feat)
        #     elif self.dep_dim and which_feat(feat_type) == 'dep':
        #         embed_feat = self.dep_embeddings(feat)
        #     else:
        #         continue
        #
        #     if self.params['mention_cat'] == 'sum':
        #         embed_feat = embed_feat.sum(dim=1)
        #     elif self.params['mention_cat'] == 'max':
        #         embed_feat = embed_feat.max(dim=1)[0]
        #     elif self.params['mention_cat'] == 'mean':
        #         embed_feat = embed_feat.mean(dim=1)
        #
        #     if which_branch(feat_type) == 'sour':
        #         feat_input.append(embed_feat)
        #     elif which_branch(feat_type) == 'targ':
        #         feat_input.append(embed_feat)
        #
        # cat_input.append(self.feat_drop(torch.cat(feat_input, dim=1)))

        fc_input = torch.cat(cat_input, dim=1)
        # print(fc_input.shape)

        ## FC and softmax layer
        fc1_out = F.relu(self.fc1(fc_input))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        return fc2_out


class TempBranchRNN(nn.Module):

    def __init__(self, wvocab_size, cvocab_size, pos_size, dep_size, dist_size, action_size,
                 max_sent_len, max_seq_len, max_mention_len, max_word_len,
                 pre_embed=None,
                 verbose=0, **params):

        super(TempBranchRNN, self).__init__()
        self.params = params
        self.hidden_dim = self.params['seq_rnn_dim']
        self.sent_hidden_dim = self.params['sent_rnn_dim']
        self.max_sent_len = max_sent_len
        self.max_seq_len = max_seq_len
        self.max_mention_len = max_mention_len
        self.max_word_len = max_word_len
        self.verbose = verbose

        if isinstance(pre_embed, np.ndarray):
            self.word_dim = pre_embed.shape[1]
            self.word_embeddings = TempUtils.pre2embed(pre_embed)

        self.char_dim = params['char_dim']
        if self.char_dim:
            self.char_embeddings = nn.Embedding(cvocab_size, self.char_dim, padding_idx=0)
            self.char_hidden_dim = params['char_dim']
            self.char_rnn = nn.LSTM(self.char_dim, self.char_hidden_dim // 2,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True)

        self.pos_dim = params['pos_dim']
        if self.pos_dim:
            self.pos_embeddings = nn.Embedding(pos_size, self.pos_dim, padding_idx=0)

        self.dep_dim = params['dep_dim']
        if self.dep_dim:
            self.dep_embeddings = nn.Embedding(dep_size, self.dep_dim, padding_idx=0)

        self.dist_dim = params['dist_dim']
        if self.dist_dim:
            self.dist_embeddings = nn.Embedding(dist_size, self.dist_dim, padding_idx=0)

        self.sent_input_dim = self.word_dim + self.char_dim + 2 * self.dist_dim

        self.sent_rnn = nn.LSTM(self.sent_input_dim,
                                self.sent_hidden_dim // 2,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

        if self.params['sent_rnn_pool']:
            self.sent_rnn_pool = nn.MaxPool1d(self.max_sent_len)

        self.seq_input_dim = seq_input_dim(self.params, self.word_dim)

        self.sour_rnn = nn.LSTM(self.seq_input_dim,
                                self.hidden_dim // 2,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

        if self.params['link_type'] not in ['Event-DCT']:
            self.targ_rnn = nn.LSTM(self.seq_input_dim,
                                    self.hidden_dim // 2,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True)

        if self.params['seq_rnn_pool']:
            self.seq_rnn_pool = nn.MaxPool1d(self.max_seq_len)

        self.sour_rnn_drop = nn.Dropout(p=self.params['dropout_sour_rnn'])
        if self.params['link_type'] not in ['Event-DCT']:
            self.targ_rnn_drop = nn.Dropout(p=self.params['dropout_targ_rnn'])

        self.feat_drop = nn.Dropout(p=self.params['dropout_feat'])

        self.fc1_input_dim = self.hidden_dim + self.sent_hidden_dim + self.word_dim + self.pos_dim + self.dep_dim

        if self.params['link_type'] not in ['Event-DCT']:
            self.fc1_input_dim *= 2

        self.fc1 = nn.Linear(self.fc1_input_dim, self.params['fc_hidden_dim'])
        self.fc1_drop = nn.Dropout(p=self.params['dropout_fc'])
        self.fc2 = nn.Linear(self.params['fc_hidden_dim'], action_size)

    def init_hidden(self, batch_size, hidden_dim):
        return (torch.zeros(2, batch_size, hidden_dim // 2).to(device),
                torch.zeros(2, batch_size, hidden_dim // 2).to(device))

    def forward(self, **input_dict):

        batch_size = input_dict['sour_word_seq'].shape[0]


        """
        sour/targ seq rnn 
        """
        sour_seq_input, targ_seq_input = [], []

        for feat_type, feat in {k: v for k, v in input_dict.items() if is_feat(k) and is_seq_feat(k)}.items():

            if self.word_dim and which_feat(feat_type) == 'word':
                embed_feat = self.word_embeddings(feat)
            elif self.pos_dim and which_feat(feat_type) == 'pos':
                embed_feat = self.pos_embeddings(feat)
            elif self.dep_dim and which_feat(feat_type) == 'dep':
                embed_feat = self.dep_embeddings(feat)
            elif self.char_dim and which_feat(feat_type) == 'char':
                embed_char = self.char_embeddings(feat.view(batch_size * self.max_seq_len, self.max_word_len))
                self.char_hidden = self.init_hidden(batch_size * self.max_seq_len, self.char_hidden_dim)
                char_outs, self.char_hidden = self.char_rnn(embed_char, self.char_hidden)
                embed_feat = char_outs[:, -1, :].view((batch_size, self.max_seq_len, -1))
            else:
                continue

            if which_branch(feat_type) == 'sour':
                sour_seq_input.append(embed_feat)
            elif which_branch(feat_type) == 'targ':
                targ_seq_input.append(embed_feat)

        self.sour_rnn_hidden = self.init_hidden(batch_size, self.hidden_dim)
        self.targ_rnn_hidden = self.init_hidden(batch_size, self.hidden_dim)

        sour_seq_tensor = torch.cat(sour_seq_input, dim=2)
        sour_rnn_out, self.sour_rnn_hidden = self.sour_rnn(sour_seq_tensor, self.sour_rnn_hidden)

        if targ_seq_input:
            targ_seq_tensor = torch.cat(targ_seq_input, dim=2)
            targ_rnn_out, self.targ_rnn_hidden = self.targ_rnn(targ_seq_tensor, self.targ_rnn_hidden)

        if self.params['seq_rnn_pool']:
            sour_rnn_out = self.seq_rnn_pool(sour_rnn_out.transpose(1, 2)).squeeze()
            if targ_seq_input:
                targ_rnn_out = self.seq_rnn_pool(targ_rnn_out.transpose(1, 2)).squeeze()
        else:
            sour_rnn_out = sour_rnn_out[:, -1, :]
            if targ_seq_input:
                targ_rnn_out = targ_rnn_out[:, -1, :]

        """
        full sent rnn 
        """
        sent_input = []
        for feat_type, feat in {k: v for k, v in input_dict.items() if is_feat(k) and is_sent_feat(k)}.items():
            if self.word_dim and which_feat(feat_type) == 'word':
                embed_feat = self.word_embeddings(feat)
            elif self.dist_dim and which_feat(feat_type) == 'dist':
                print(feat.shape)
                embed_feat = self.dist_embeddings(feat)
                # print(embed_feat.shape)
            elif self.char_dim and which_feat(feat_type) == 'char':
                embed_char = self.char_embeddings(feat.view(batch_size * self.max_sent_len, self.max_word_len))
                self.char_hidden = self.init_hidden(batch_size * self.max_sent_len, self.char_hidden_dim)
                char_outs, self.char_hidden = self.char_rnn(embed_char, self.char_hidden)
                embed_feat = char_outs[:, -1, :].view((batch_size, self.max_seq_len, -1))
            sent_input.append(embed_feat)

        sent_tensor = torch.cat(sent_input, dim=2)
        self.sent_rnn_hidden = self.init_hidden(batch_size, self.sent_hidden_dim)
        sent_rnn_out, self.sent_rnn_hidden = self.sent_rnn(sent_tensor, self.sent_rnn_hidden)

        if self.params['sent_rnn_pool']:
            sent_rnn_out = self.sent_rnn_pool(sent_rnn_out.transpose(1, 2)).squeeze()
        else:
            sent_rnn_out = sent_rnn_out[:, -1, :]


        """
        concatenate seq rnn + sent rnn + feat
        """
        cat_input = []

        cat_input.append(self.sour_rnn_drop(sour_rnn_out))
        if targ_seq_input:
            cat_input.append(self.targ_rnn_drop(targ_rnn_out))

        cat_input.append(sent_rnn_out)

        feat_input = []
        for feat_type, feat in {k: v for k, v in input_dict.items() if is_feat(k) and is_tok_feat(k)}.items():

            if self.word_dim and which_feat(feat_type) == 'word':
                embed_feat = self.word_embeddings(feat)
            elif self.pos_dim and which_feat(feat_type) == 'pos':
                embed_feat = self.pos_embeddings(feat)
            elif self.dep_dim and which_feat(feat_type) == 'dep':
                embed_feat = self.dep_embeddings(feat)
            else:
                continue

            if self.params['mention_cat'] == 'sum':
                embed_feat = embed_feat.sum(dim=1)
            elif self.params['mention_cat'] == 'max':
                embed_feat = embed_feat.max(dim=1)[0]
            elif self.params['mention_cat'] == 'mean':
                embed_feat = embed_feat.mean(dim=1)

            if which_branch(feat_type) == 'sour':
                feat_input.append(embed_feat)
            elif which_branch(feat_type) == 'targ':
                feat_input.append(embed_feat)

        cat_input.append(self.feat_drop(torch.cat(feat_input, dim=1)))

        fc_input = torch.cat(cat_input, dim=1)

        ## FC and softmax layer
        fc1_out = F.relu(self.fc1(fc_input))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        return fc2_out


class TempRNN(nn.Module):

    def __init__(self, seq_len, action_size, verbose=0, **params):
        super(TempRNN, self).__init__()

        ## parameters
        self.hidden_dim = params['filter_nb']
        self.batch_size = params['batch_size']
        self.verbose_level = verbose

        ## neural layers
        self.embedding_dropout = nn.Dropout(p=params['dropout_emb'])
        rnn1_input_dim = 0
        for feat_type in feat_types:
            if feat_type.split('_')[-1] == 'seq':
                if feat_type.split('_')[-2] == 'token':
                    rnn1_input_dim += params['word_dim']
                elif feat_type.split('_')[-2] == 'dist':
                    rnn1_input_dim += params['pos_dim']
        self.rnn1 = nn.LSTM(rnn1_input_dim, params['filter_nb'] // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.rnn1hid_drop = nn.Dropout(p=params['dropout_rnn'])

        self.cat_drop = nn.Dropout(p=params['dropout_cat'])
        fc1_input_dim = params['filter_nb']
        for feat_type in feat_types:
            if feat_type.split('_')[-1] == 'token':
                fc1_input_dim += params['word_dim']
            elif feat_type.split('_')[-1] == 'dist':
                fc1_input_dim += params['pos_dim']

        self.fc1 = nn.Linear(fc1_input_dim, params['fc_hidden_dim'])
        self.fc1_drop = nn.Dropout(p=params['dropout_fc'])
        self.fc2 = nn.Linear(params['fc_hidden_dim'], action_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.hidden_dim // 2).to(device),
                torch.zeros(2, batch_size, self.hidden_dim // 2).to(device))

    def forward(self, feat_types, *feat_inputs):

        ## RNN1 network

        seq_inputs = []

        for feat, feat_type in zip(feat_inputs, feat_types):
            if feat_type.split('_')[-1] == 'seq':
                if self.verbose_level:
                    print(feat_type, feat.shape)
                seq_inputs.append(feat)
        seq_inputs = torch.cat(seq_inputs, dim=2)
        embed_inputs = self.embedding_dropout(seq_inputs) # [batch, len, dim]

        if self.verbose_level:
            print("embedd_input size:", embed_inputs.shape)

        self.rnn1_hidden = self.init_hidden(embed_inputs.shape[0])
        rnn1_out, self.rnn1_hidden = self.rnn1(embed_inputs, self.rnn1_hidden)
        if self.verbose_level:
            print("rnn_out size:", rnn1_out.shape, "position_input,", position_input.shape)

        cat_inputs = [rnn1_out[-1]]
        for feat, feat_type in zip(feat_inputs, feat_types):
            if feat_type.split('_')[-1] != 'seq':
                if self.verbose_level:
                    print(feat_type, feat.shape)
                    print('tok pool:', self.tok_p1(feat).squeeze(-1).shape)
                cat_inputs.append(self.tok_p1(feat).squeeze(-1))
        cat_out = torch.cat(cat_inputs, dim=1)
        cat_out = self.cat_dropout(cat_out)
        if self.verbose_level:
            print("cat_output size:", cat_out.shape)

        ## FC and softmax layer
        fc1_out = F.relu(self.fc1(cat_out))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        return fc2_out


class TempAttnRNN(nn.Module):

    def __init__(self, vocab_size, pos_size, action_size, max_len, pre_model=None, **params):
        super(TempAttnRNN, self).__init__()
        self.hidden_dim = params['hidden_dim']
        self.rnn1 = nn.LSTM(params['word_dim'] + params['pos_dim'], params['hidden_dim'] // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.rnn1_hidden = self.init_hidden()
        self.rnn1hid_drop = nn.Dropout(p=params['dropout_rnn'])
        self.attn_W = torch.randn(params['hidden_dim'], requires_grad=True)
        self.cat_drop = nn.Dropout(p=params['dropout_cat'])
        self.fc1 = nn.Linear(params['filter_nb'], params['fc_hidden_dim'])
        self.fc1_drop = nn.Dropout(p=params['dropout_fc'])
        self.fc2 = nn.Linear(params['fc_hidden_dim'], action_size)

    def init_hidden(self):
        return (torch.zeros(2, self.batch_size, self.hidden_dim // 2),
                torch.zeros(2, self.batch_size, self.hidden_dim // 2))

    def forward(self, word_input, position_input):

        ## RNN1 network
        cat_input = torch.cat((word_input, position_input), dim=2)
        rnn1_out, self.rnn1_hidden = self.rnn1(cat_input, self.rnn1hid_drop(self.hidden_dim))

        ## Attention layer
        attn_in = rnn1_out.transpose(1, 2) # transpose rnn1_out from [batch_size, max_len, hidden_dim] to [batch_size, hidden_dim, max_len]
        attn_M = F.tanh(attn_in)
        W = self.attn_W.unsqueeze(0).expand(10, -1, -1)   # W: [batch_size, 1, hidden_dim]
        attn_alpha = F.softmax(torch.bmm(W, attn_M), dim=2)       # rnn1_alpha: [batch_size, 1, max_len]
        attn_out = torch.bmm(attn_in, attn_alpha.transpose(1, 2)) # attn_out: [batch_size, hidden_dim, 1]

        ## catenate the RNN1 output and position embeddings
        cat_out = torch.cat((attn_out.squeeze(), position_input[:, 0, :], position_input[:, -1, :]), dim=1)
        cat_out = self.cat_drop(cat_out)

        ## FC and softmax layer
        fc1_out = F.relu(self.fc1(p1_out))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        return fc2_out


class TempClassifier(nn.Module):

    def __init__(self, wvocab_size, cvocab_size, pos_size, action_size,
                 max_seq_len, max_token_len, max_char_len,
                 feat_types,
                 pre_model=None,
                 verbose_level=1,
                 **params):
        super(TempClassifier, self).__init__()
        self.classifier = params['classifier']
        self.max_seq_len = max_seq_len
        self.max_token_len = max_token_len
        self.max_char_len = max_char_len
        self.verbose_level = verbose_level

        if params['char_emb']:
            self.char_embeddings = nn.Embedding(cvocab_size, params['char_dim'], padding_idx=0)
            self.char_dim = params['char_dim']
            self.char_hidden_dim = params['char_dim']
            self.char_rnn = nn.LSTM(self.char_dim, self.char_hidden_dim // 2,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True)


        if isinstance(pre_model, np.ndarray):
            self.word_embeddings = TempUtils.pre2embed(pre_model)
        else:
            self.word_embeddings = nn.Embedding(wvocab_size, params['word_dim'], padding_idx=0)
        self.position_embeddings = nn.Embedding(pos_size, params['pos_dim'], padding_idx=0)

        if self.classifier == 'CNN':
            self.temp_detector = TempCNN(max_seq_len, max_token_len, max_char_len, action_size, feat_types, verbose_level=self.verbose_level, **params)
        elif self.classifier == 'AttnCNN':
            self.temp_detector = TempAttnCNN(max_seq_len, max_token_len, max_char_len, action_size, feat_types, verbose_level=self.verbose_level, **params)
        elif self.classifier == 'RNN':
            self.temp_detector = TempRNN(max_seq_len, max_token_len, max_char_len, action_size, feat_types, verbose_level=self.verbose_level, **params)
        elif self.classifier == 'AttnRNN':
            self.temp_detector = TempAttnRNN(max_seq_len, max_token_len, max_char_len, action_size, feat_types, **params)
        else:
            raise Exception("[ERROR]Wrong classifier param [%s] selected...." % (self.classifier) )

    def init_char_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.char_hidden_dim // 2).to(device),
                torch.zeros(2, batch_size, self.char_hidden_dim // 2).to(device))

    def forward(self, feat_types, *feat_inputs, **params):

        assert(len(feat_inputs) == len(feat_types))

        embedded_inputs = []

        batch_size, max_len = feat_inputs[0].size()

        for feat, feat_type in zip(feat_inputs, feat_types):
            if feat_type in ['word_seq']:
                # print(feat[0])
                # print(self.word_embeddings(feat)[0])
                embedded_inputs.append(self.word_embeddings(feat))
            if feat_type in ['char_seq']:
                if params['char_emb']:
                    embedded_feat = self.char_embeddings(feat.view(batch_size * self.max_seq_len, self.max_char_len))
                    self.char_hidden = self.init_char_hidden(batch_size * self.max_seq_len)
                    char_outs, self.char_hidden = self.char_rnn(embedded_feat, self.char_hidden)
                    char_out = char_outs[:, -1, :].view((batch_size, self.max_seq_len, -1))
                    embedded_inputs.append(char_out)
                else:
                    embedded_inputs.append(None)
            elif feat_type in ['sour_dist_seq', 'targ_dist_seq']:
                embedded_inputs.append(self.position_embeddings(feat))
            elif feat_type in ['sour_word_tok', 'targ_word_tok']:
                embedded_inputs.append(self.word_embeddings(feat))
            elif feat_type in ['sour_dist_tok', 'targ_dist_tok']:
                embedded_inputs.append(self.position_embeddings(feat))

        if self.verbose_level == 2:
            print("number of feat types:", len(embedded_inputs))

        temp_out = self.temp_detector(feat_types, *embedded_inputs, **params)

        return temp_out


class TimexCNN(nn.Module):
    def __init__(self, input_dim, c1_dim, seq_len, fc1_dim, action_size, window):
        super(TimexCNN, self).__init__()
        self.cl1 = nn.Conv1d(input_dim, c1_dim, window)
        self.pl1 = nn.MaxPool1d(seq_len - window + 1)
        self.cl2 = nn.Conv1d(input_dim, c1_dim, window)
        self.pl2 = nn.MaxPool1d(seq_len - window + 1)
        self.fc1 = nn.Linear(c1_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, action_size)

    def forward(self, left_input, right_input):
        ## dct_input (batch_size, seq_len, input_dim)
        cl1_out = F.relu(self.cl1(left_input.transpose(1, 2)))
        pl1_out = F.dropout(self.p1(cl1_out))
        cr1_out = F.relu(self.cr1(right_input.transpose(1, 2)))
        pr1_out = F.dropout(self.p1(cr1_out))

        cat_out = torch.cat((pl1_out, pr1_out), 1)
        fc1_out = F.relu(self.fc1(cat_out))
        fc2_out = F.log_softmax(fc1_out, dim=1)
        return fc2_out


class DCTDetector(nn.Module):
    def __init__(self, embedding_dim, dct_hidden_dim, action_size, batch_size):
        super(DCTDetector, self).__init__()
        ## initialize parameters
        self.dct_hidden_dim = dct_hidden_dim
        self.batch_size = batch_size

        ## initialize neural layers
        self.dct_tagger = nn.LSTM(embedding_dim, dct_hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.tagger_dropout = nn.Dropout(p=0.5)
        self.dct_hidden2action = nn.Linear(dct_hidden_dim, action_size)
        self.dct_hidden = self.init_dct_hidden()

    def init_dct_hidden(self):
        return (torch.zeros(2, self.batch_size, self.dct_hidden_dim // 2, device=device),
                torch.zeros(2, self.batch_size, self.dct_hidden_dim // 2, device=device))

    def forward(self, dct_embeds):
        # print(dct_embeds)
        self.init_dct_hidden()
        dct_out, self.dct_hidden = self.dct_tagger(dct_embeds, self.dct_hidden)
        dct_out = self.dct_hidden2action(dct_out[:, -1, :])
        # print(dct_out)
        dct_score = F.log_softmax(dct_out, dim=1)
        # print(dct_score)
        return dct_score

class TimeDetector(nn.Module):

    def __init__(self, embedding_dim, time_hidden_dim, action_size, batch_size):
        super(TimeDetector, self).__init__()
        ## initialize parameters
        self.time_hidden_dim = time_hidden_dim
        self.batch_size = batch_size

        ## initialize neural layers
        self.left_tagger = nn.LSTM(embedding_dim, time_hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.left_tagger_dropout = nn.Dropout(p=0.5)
        self.right_tagger = nn.LSTM(embedding_dim, time_hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.right_tagger_dropout = nn.Dropout(p=0.5)
        self.time_hidden2action = nn.Linear(time_hidden_dim * 2, action_size)
        self.left_hidden = self.init_left_time_hidden()
        self.right_hidden = self.init_right_time_hidden()

    def init_left_time_hidden(self):
        return (torch.zeros(2, self.batch_size, self.time_hidden_dim // 2, device=device),
                torch.zeros(2, self.batch_size, self.time_hidden_dim // 2, device=device))

    def init_right_time_hidden(self):
        return (torch.zeros(2, self.batch_size, self.time_hidden_dim // 2, device=device),
                torch.zeros(2, self.batch_size, self.time_hidden_dim // 2, device=device))

    def forward(self, left_embeds, right_embeds):

        ## left branch
        left_out, self.left_hidden = self.left_tagger(left_embeds, self.left_hidden)
        left_out = self.left_tagger_dropout(left_out[:,-1, :])

        ## right branch
        right_out, self.right_hidden = self.right_tagger(right_embeds, self.right_hidden)
        right_out = self.right_tagger_dropout(right_out[:,-1, :])

        ## concatenation

        time_out = torch.cat((left_out, right_out), 1)
        time_out = self.time_hidden2action(time_out)
        time_score = F.log_softmax(time_out, dim=1)
        # print(time_score)
        return time_score


class DqnInferrer(nn.Module):
    def __init__(self, embedding_dim, dct_hidden_dim, time_hidden_dim, vocab_size, action_size, batch_size):
        super(DqnInferrer, self).__init__()
        self.dct_hidden_dim = dct_hidden_dim
        self.time_hidden_dim = time_hidden_dim
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dct_embedding_dropout = nn.Dropout(p=0.5)
        self.left_embedding_dropout = nn.Dropout(p=0.5)
        self.right_embedding_dropout = nn.Dropout(p=0.5)
        self.dct_detector = DCTDetector(self.embedding_dim,
                                        self.dct_hidden_dim,
                                        action_size,
                                        batch_size)
        self.time_detector = TimeDetector(self.embedding_dim,
                                          self.time_hidden_dim,
                                          action_size,
                                          batch_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, state_i, dpath_input):
        if state_i == 0:
            dct_input = dpath_input.view(self.batch_size, -1)
            dct_embeds = self.word_embeddings(dct_input).view(self.batch_size, dct_input.size()[-1], -1)
            dct_embeds = self.dct_embedding_dropout(dct_embeds)
            self.dct_detector.init_dct_hidden()
            dct_score = self.dct_detector(dct_embeds)
            # print(dct_score)
            return dct_score
        else:
            ## embed input of the left branch
            left_input = dpath_input[:, 0, :]
            left_embeds = self.word_embeddings(left_input).view(self.batch_size, left_input.size()[-1], -1)
            left_embeds = self.left_embedding_dropout(left_embeds)

            ## embed input of the right branch
            right_input = dpath_input[:, 1, :]
            right_embeds = self.word_embeddings(right_input).view(self.batch_size, right_input.size()[-1], -1)
            right_embeds = self.right_embedding_dropout(right_embeds)

            self.time_detector.init_left_time_hidden()
            self.time_detector.init_right_time_hidden()
            time_score = self.time_detector(left_embeds, right_embeds).view(self.batch_size, 1, -1)
            #             print(time_score.size())
            # print(time_score)
            return time_score.squeeze(0)

class TimeInferrer(nn.Module):

    def __init__(self, embedding_dim, dct_hidden_dim, time_hidden_dim, vocab_size, action_size, batch_size):
        super(TimeInferrer, self).__init__()
        self.dct_hidden_dim = dct_hidden_dim
        self.time_hidden_dim = time_hidden_dim
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dct_embedding_dropout = nn.Dropout(p=0.5)
        self.left_embedding_dropout = nn.Dropout(p=0.5)
        self.right_embedding_dropout = nn.Dropout(p=0.5)
        self.dct_detector = DCTDetector(self.embedding_dim,
                                        self.dct_hidden_dim,
                                        action_size,
                                        batch_size)
        self.time_detector = TimeDetector(self.embedding_dim,
                                          self.time_hidden_dim,
                                          action_size,
                                          batch_size)

    def forward(self, dct_input, time_inputs):

        ## event to dct
        dct_var = dct_input.view(self.batch_size, -1)
        dct_embeds = self.word_embeddings(dct_var).view(self.batch_size, dct_input.size()[-1], -1)
        dct_embeds = self.dct_embedding_dropout(dct_embeds)
        dct_score = self.dct_detector(dct_embeds)
        out_scores = dct_score.clone().view(self.batch_size, 1, -1)
#         print(out_scores.size())

        ## event to per time expression
        for time_index in range(time_inputs.size()[1]):

            ## embed input of the left branch
            left_input = time_inputs[:, time_index, 0, :]
            left_embeds = self.word_embeddings(left_input).view(self.batch_size, left_input.size()[-1], -1)
            left_embeds = self.left_embedding_dropout(left_embeds)

            ## embed input of the right branch
            right_input = time_inputs[:, time_index, 1, :]
            right_embeds = self.word_embeddings(right_input).view(self.batch_size, right_input.size()[-1], -1)
            right_embeds = self.right_embedding_dropout(right_embeds)

            time_score = self.time_detector(left_embeds, right_embeds).view(self.batch_size, 1, -1)
#             print(time_score.size())
            out_scores = torch.cat((out_scores, time_score), dim=1)
#             print(out_scores.size())


        return out_scores


def score2pred_2(out_scores, norm_times, BATCH_SIZE, update_strategies):
    norm_times = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long)
    preds_out = torch.ones((BATCH_SIZE, 3), dtype=torch.long) * -1
    preds_out.requires_grad_()
    class_ids = torch.argmax(out_scores, dim=2)
    for i in range(out_scores.size()[0]):
        for cid, time in zip(class_ids[i], norm_times):
            preds_out[i] = update_strategies[cid] * time + (torch.ones_like(update_strategies[cid], dtype=torch.long) - update_strategies[cid]) * preds_out[i]
    return preds_out

def diff_elem_loss(pred_out, target):
    diff_t = torch.eq(pred_out, target)
    loss = torch.div((pred_out.numel() - diff_t.sum().float()), pred_out.numel())
    loss.requires_grad_()
    return loss

def new_loss(out_scores, target, BATCH_SIZE, update_strategies):
    norm_times = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long)
    pred_out = torch.ones((BATCH_SIZE, 3), dtype=torch.long) * -1
    # pred_out.requires_grad_()
    class_ids = torch.argmax(out_scores, dim=2)
    for i in range(out_scores.size()[0]):
        for cid, time in zip(class_ids[i], norm_times):
            pred_out[i] = update_strategies[cid] * time + (
                        torch.ones_like(update_strategies[cid], dtype=torch.long) - update_strategies[cid]) * pred_out[i]
    diff_t = torch.eq(pred_out, target)
    loss = torch.div((pred_out.numel() - diff_t.sum().float()), pred_out.numel())
    loss.requires_grad_()
    return loss

def main():

    EMBEDDING_DIM = 64
    DCT_HIDDEN_DIM = 60
    TIME_HIDDEN_DIM = 50
    VOCAB_SIZE = 100
    ACTION_SIZE = 8
    EPOCH_NUM = 5
    learning_rate = 0.01

    model = TimeInferrer(EMBEDDING_DIM, DCT_HIDDEN_DIM, TIME_HIDDEN_DIM, VOCAB_SIZE, ACTION_SIZE, BATCH_SIZE)

    optimizer = optim.RMSprop(model.parameters())
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


    ## train model
    a = list(model.parameters())[1]

    for epoch in range(EPOCH_NUM):

        total_loss = torch.Tensor([0])
        for step, (dct_input, time_input, target) in enumerate(loader):


            model.zero_grad()

            BATCH=dct_input.size()[0]

            time_scores = model(dct_input, time_input)
    #         pred_out = score2pred_2(time_scores, None, BATCH, update_strategies)
    #         print(time_scores.size(), pred_out.size(), target.size())
    # #         print(pred_out.requires_grad)
    #         loss = diff_elem_loss(pred_out, target)

            loss = new_loss(time_scores, target, BATCH, update_strategies)
            print(time_scores.size(), target.size())

            loss.backward(retain_graph=True)
            optimizer.step()
            ## check if the model parameters being updated

    #         print(len(model.word_embeddings.weight[1]))
    #         print(list(model.parameters())[0].grad)


            total_loss += loss.data.item() * BATCH
    #         print('')
        for name, param in model.named_parameters():
            if name == 'time_detector.time_hidden2action.weight':
                print(param[0][:5])
        # print(model.word_embeddings(torch.LongTensor([[0, ]])).squeeze()[:5])
        b = list(model.parameters())[1]
        print('Epoch', epoch, 'Step', step, ', Epoch loss: %.4f' % (total_loss / 500), torch.equal(a.data, b.data))
    #     print(model.word_embeddings.data)

if __name__ == "__main__":
    main()
