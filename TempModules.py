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


class TempCNN(nn.Module):

    def __init__(self, seq_len, token_len, action_size, verbose_level=0, **params):
        super(TempCNN, self).__init__()
        self.verbose_level = verbose_level
        self.embedding_dropout = nn.Dropout(p=params['dropout_emb'])
        self.c1 = nn.Conv1d(params['word_dim'] + 2 * params['pos_dim'], params['filter_nb'], params['kernel_len'])
        self.p1 = nn.MaxPool1d(seq_len - params['kernel_len'] + 1)
        self.tok_p1 = nn.MaxPool1d(token_len)
        self.cat_dropout = nn.Dropout(p=params['dropout_cat'])
        self.fc1 = nn.Linear(params['filter_nb'] + 2 * params['word_dim'] + 2 * params['pos_dim'], params['fc_hidden_dim'])
        self.fc1_drop = nn.Dropout(p=params['dropout_fc'])
        self.fc2 = nn.Linear(params['fc_hidden_dim'], action_size)

    def forward(self, feat_types, *feat_inputs):

        ## input (batch_size, seq_len, input_dim) => (batch_size, input_dim, seq_len)

        seq_inputs = []

        for feat, feat_type in zip(feat_inputs, feat_types):
            if feat_type.split('_')[-1] == 'seq':
                if self.verbose_level:
                    print(feat_type, feat.shape)
                seq_inputs.append(feat)
        seq_inputs = torch.cat(seq_inputs, dim=2).transpose(1, 2)
        embed_inputs = self.embedding_dropout(seq_inputs)

        c1_out = F.relu(self.c1(embed_inputs))

        if self.verbose_level:
            print("c1_output size:", c1_out.shape)

        p1_out = self.p1(c1_out).squeeze(-1)

        if self.verbose_level:
            print("p1_output size:", p1_out.shape)


        cat_inputs = [p1_out]
        for feat, feat_type in zip(feat_inputs, feat_types):
            if feat_type.split('_')[-1] != 'seq':
                if self.verbose_level:
                    print(feat_type, feat.shape)
                    print('tok pool:', self.tok_p1(feat.transpose(1, 2)).squeeze(-1).shape)
                cat_inputs.append(self.tok_p1(feat.transpose(1, 2)).squeeze(-1))
        cat_out = torch.cat(cat_inputs, dim=1)
        cat_out = self.cat_dropout(cat_out)
        if self.verbose_level:
            print("cat_output size:", cat_out.shape)

        fc1_out = F.relu(self.fc1(cat_out))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        if self.verbose_level:
            print()

        return fc2_out


class TempAttnCNN(nn.Module):

    def __init__(self, seq_len, token_len, action_size, verbose=0, **params):
        super(TempAttnCNN, self).__init__()
        self.verbose_level = verbose
        self.embedding_dropout = nn.Dropout(p=params['dropout_emb'])
        self.c1 = nn.Conv1d(params['word_dim'] + 2 * params['pos_dim'], params['filter_nb'], params['kernel_len'])
        self.attn_W = torch.nn.Parameter(torch.randn(params['filter_nb'], requires_grad=True))
        self.tok_p1 = nn.MaxPool1d(token_len)
        self.cat_dropout = nn.Dropout(p=params['dropout_cat'])
        self.fc1 = nn.Linear(params['filter_nb'] + 2 * params['word_dim'] + 2 * params['pos_dim'], params['fc_hidden_dim'])
        self.fc1_drop = nn.Dropout(p=params['dropout_fc'])
        self.fc2 = nn.Linear(params['fc_hidden_dim'], action_size)

    def forward(self, feat_types, *feat_inputs):

        seq_inputs = []

        for feat, feat_type in zip(feat_inputs, feat_types):
            if feat_type.split('_')[-1] == 'seq':
                if self.verbose_level:
                    print(feat_type, feat.shape)
                seq_inputs.append(feat)
        ## input (batch_size, seq_len, input_dim) => (batch_size, input_dim, seq_len)
        seq_inputs = torch.cat(seq_inputs, dim=2).transpose(1, 2)
        embed_inputs = self.embedding_dropout(seq_inputs)


        batch_size = embed_inputs.shape[0]
        if self.verbose_level:
            print("embed_input size:", embed_inputs.shape)

        c1_out = F.relu(self.c1(embed_inputs))
        if self.verbose_level:
            print("c1_output size:", c1_out.shape)

        attn_M = F.tanh(c1_out)  # attn_M: [batch, filter_nb, kernel_out]
        W = self.attn_W.unsqueeze(0).expand(batch_size, -1, -1)  # W: [batch_size, 1, filter_nb]
        attn_alpha = F.softmax(torch.bmm(W, attn_M), dim=2)  # rnn1_alpha: [batch_size, 1, kernel_out]
        attn_out = F.tanh(torch.bmm(c1_out, attn_alpha.transpose(1, 2)))  # attn_out: [batch_size, filter_nb, 1]

        # p1_out = self.p1(c1_out).squeeze(-1)

        # if self.verbose_level:
        #     print("p1_output size:", p1_out.shape)

        cat_inputs = [attn_out.squeeze()]
        for feat, feat_type in zip(feat_inputs, feat_types):
            if feat_type.split('_')[-1] != 'seq':
                if self.verbose_level:
                    print(feat_type, feat.shape)
                    print('tok pool:', self.tok_p1(feat.transpose(1, 2)).squeeze(-1).shape)
                cat_inputs.append(self.tok_p1(feat.transpose(1, 2)).squeeze(-1))
        cat_out = torch.cat(cat_inputs, dim=1)
        cat_out = self.cat_dropout(cat_out)
        # if self.verbose_level:
        #     print("cat_output size:", cat_out.shape)

        fc1_out = F.relu(self.fc1(cat_out))
        fc1_out = self.fc1_drop(fc1_out)
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        if self.verbose_level:
            print()

        # print(self.attn_W[:5])

        return fc2_out

class TempRNN(nn.Module):

    def __init__(self, seq_len, action_size, verbose=0, **params):
        super(TempRNN, self).__init__()

        ## parameters
        self.hidden_dim = params['filter_nb']
        self.batch_size = params['batch_size']
        self.verbose_level = verbose

        ## neural layers
        self.rnn1 = nn.LSTM(params['word_dim'] + 2 * params['pos_dim'], params['filter_nb'] // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.rnn1hid_drop = nn.Dropout(p=params['dropout_rnn'])
        self.cat_drop = nn.Dropout(p=params['dropout_cat'])
        self.fc1 = nn.Linear(params['filter_nb'] + 4 * params['pos_dim'], params['fc_hidden_dim'])
        self.fc1_drop = nn.Dropout(p=params['dropout_fc'])
        self.fc2 = nn.Linear(params['fc_hidden_dim'], action_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.hidden_dim // 2).to(device),
                torch.zeros(2, batch_size, self.hidden_dim // 2).to(device))

    def forward(self, word_input, position_input):

        ## RNN1 network
        cat_input = torch.cat((word_input, position_input), dim=2) # [batch, len, dim]
        if self.verbose_level:
            print("cat_input size:", cat_input.shape)
        # cat_input = cat_input.transpose(0, 1) # from [batch, len, dim] to [len, batch, dim]

        self.rnn1_hidden = self.init_hidden(cat_input.shape[0])
        rnn1_out, self.rnn1_hidden = self.rnn1(cat_input, self.rnn1_hidden)
        if self.verbose_level:
            print("rnn_out size:", rnn1_out.shape, "position_input,", position_input.shape)

        ## catenate the RNN1 output and position embeddings: hidden_dim + 2 * 2 * pos_dim
        cat_out = torch.cat((rnn1_out[:, -1, :], position_input[:, 0, :], position_input[:, -1, :]), dim=1)
        if self.verbose_level:
            print("cat_out shape:", cat_out.shape)
        cat_out = self.cat_drop(cat_out)

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

    def __init__(self, vocab_size, pos_size, action_size, max_len, max_token_len, feat_types, pre_model=None, verbose_level=0, **params):
        super(TempClassifier, self).__init__()
        self.classifier = params['classifier']
        self.verbose_level = verbose_level

        if isinstance(pre_model, np.ndarray):
            self.word_embeddings = TempUtils.pre2embed(pre_model)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, params['word_dim'], padding_idx=0)
        self.position_embeddings = nn.Embedding(pos_size, params['pos_dim'], padding_idx=0)

        if self.classifier == 'CNN':
            self.temp_detector = TempCNN(max_len, max_token_len, action_size, verbose_level=self.verbose_level, **params)
        elif self.classifier == 'AttnCNN':
            self.temp_detector = TempAttnCNN(max_len, max_token_len, action_size, verbose_level=self.verbose_level, **params)
        elif self.classifier == 'RNN':
            self.temp_detector = TempRNN(max_len, max_token_len, action_size, verbose_level=self.verbose_level, **params)
        elif self.classifier == 'AttnRNN':
            self.temp_detector = TempAttnRNN(max_len, max_token_len, action_size, **params)
        else:
            raise Exception("[ERROR]Wrong classifier param [%s] selected...." % (self.classifier) )

    def forward(self, feat_types, *feat_inputs):
        # print(dct_input.size(), pos_in.size())

        assert(len(feat_inputs) == len(feat_types))

        embedded_inputs = []

        batch_size, max_len = feat_inputs[0].size()

        for feat, feat_type in zip(feat_inputs, feat_types):
            if feat_type in ['token_seq']:
                embedded_inputs.append(self.word_embeddings(feat.squeeze(1)))
            elif feat_type in ['sour_dist_seq', 'targ_dist_seq']:
                embedded_inputs.append(self.position_embeddings(feat.squeeze(1)))
            elif feat_type in ['sour_token', 'targ_token']:
                embedded_inputs.append(self.word_embeddings(feat.squeeze(1)))
            elif feat_type in ['sour_dist', 'targ_dist']:
                embedded_inputs.append(self.position_embeddings(feat.squeeze(1)))

        if self.verbose_level:
            print(len(embedded_inputs))

        temp_out = self.temp_detector(feat_types, *embedded_inputs)

        # if self.verbose_level:
        #     print("word_embeded input:", word_embeded.size())
        # batch_size, max_len, _ = word_embeded.size()
        #
        # pos_embeded = self.position_embeddings(pos_in.squeeze(1))
        # pos_embeded = self.embedding_dropout(pos_embeded)
        # if self.verbose_level:
        #     print("pos_embeded input:", pos_embeded.view(batch_size, max_len, -1).size())

        # temp_out = self.temp_detector(word_embeded, pos_embeded.view(batch_size, max_len, -1))
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
                        torch.ones_like(update_strategies[cid], dtype=torch.long) - update_strategies[cid]) * pred_out[
                               i]
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





