
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from IPython.display import clear_output
import numpy as np
import random
import pdb
import gensim

import TempUtils

torch.manual_seed(23214)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# STEP1: Generate Artificial Data
# 
# We randomly generate input data:
# X1: (1000 [data size] x 2 [number of branches] x 3 [number of words for per branch input]) for Event-TIMEX classifier,
# X2: (1000 [data size] x 1 [number of branches] x 3 [number of words for per branch input]) for Event-DCT classifier.  
# 
# with corresponding gold output Y:
# Y: (1000 [data size] x 2 [dimension of output tuples])

# In[71]:


class MultipleDatasets(Data.Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

dct_inputs = torch.randint(0, 100, (500, 1, 3), dtype=torch.long, device=device)
time_inputs = torch.randint(0, 100, (500, 1, 2, 3), dtype=torch.long, device=device)

targets = torch.randint(0, 2, (500, 3), dtype=torch.long, device=device)


BATCH_SIZE = 50
# EPOCH_NUM = 100

dataset = MultipleDatasets(dct_inputs, time_inputs, targets)

loader = Data.DataLoader(
    dataset = dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 1,
)

# print(time_inputs[:, :, :, :].size())
# print(time_inputs[:, :, 0, :].size())
# print(time_inputs[:, :, 1, :].size())

# x = torch.randn(3, 4)
# torch.index_select(x, 1, torch.tensor([0, 2]))

# for epoch in range(EPOCH_NUM):
#     for step, (batch_dct, batch_time, batch_y) in enumerate(loader):
#         if step > 1:
#             continue
#         print('Epoch:', epoch, '| Step:', step, '| batch dct:', batch_dct.numpy(),  '| batch time:',
#               batch_time.numpy(), '|batch y:', batch_y.numpy())

# k=1

# for dct, time, target in zip(dct_inputs[:k], time_inputs[:k], targets[:k]):
#     print(dct)
#     print(time)
#     print(target)


# In[5]:


# word2ix = {}
# embed_model = gensim.models.KeyedVectors.load_word2vec_format('/Users/fei-c/Resources/embed/giga-aacw.d200.bin', binary=True)
# # word_vectors = embed_model.wv
# for word, value in embed_model.vocab.items():
#     word2ix[word] = value.index
# print(len(word2ix))
#
#
# # In[16]:
#
#
# pretrained_weights = torch.FloatTensor(embed_model.vectors)
# print(pretrained_weights.size())
# pretrained_embed = nn.Embedding.from_pretrained(pretrained_weights, freeze=True)
# input = torch.LongTensor([[0, ]])
# print(pretrained_embed(input).squeeze().size())
# print(pretrained_embed(input).squeeze()[:6])


# In[1]:


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
            

    
class distance_loss(nn.Module):
    
    def __init__(self):
        super(distance_loss, self).__init__()
    
    def forward(self, action_out, target):
        pass



class TempCNN(nn.Module):
    def __init__(self, seq_len, action_size, **params):
        super(TempCNN, self).__init__()
        self.c1 = nn.Conv1d(params['word_dim'] + 2 * params['pos_dim'], params['filter_nb'], params['kernel_len'])
        self.p1 = nn.MaxPool1d(seq_len - params['kernel_len'] + 1)
        self.cat_dropout = nn.Dropout(p=params['dropout_cat'])
        self.fc1 = nn.Linear(params['filter_nb'], params['fc_hidden_dim'])
        self.fc2 = nn.Linear(params['fc_hidden_dim'], action_size)

    def forward(self, word_input, position_input):

        ## input (batch_size, seq_len, input_dim) => (batch_size, input_dim, seq_len)
        cat_input = torch.cat((word_input, position_input), dim=2).transpose(1, 2)

        c1_out = F.relu(self.c1(cat_input))
        p1_out = self.p1(c1_out).squeeze(-1)

        cat_out = torch.cat((p1_out, position_input[:, 0, :], position_input[:, -1, :]), dim=1)
        cat_out = self.cat_dropout(cat_out)

        fc1_out = F.relu(self.fc1(p1_out))
        fc2_out = F.log_softmax(self.fc2(fc1_out), dim=1)

        return fc2_out

class TempClassifier(nn.Module):
    def __init__(self, vocab_size, pos_size, action_size, max_len, pre_model=None, **params):
        super(TempClassifier, self).__init__()
        self.batch_size = params['batch_size']

        if isinstance(pre_model, np.ndarray):
            self.word_embeddings = TempUtils.pre2embed(pre_model)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, params['word_dim'], padding_idx=0)
        self.position_embeddings = nn.Embedding(pos_size, params['pos_dim'], padding_idx=0)
        self.embedding_dropout = nn.Dropout(p=params['dropout_emb'])
        # self.dct_detector = Tlink.DCTDetector(embedding_dim,
        #                                       dct_hidden_dim,
        #                                       action_size,
        #                                       batch_size)
        self.dct_detector = TempCNN(max_len, action_size, **params)

    def forward(self, dct_in, pos_in):
        # print(dct_input.size(), pos_in.size())

        dct_embeds = self.word_embeddings(dct_in.squeeze(1))
        dct_embeds = self.embedding_dropout(dct_embeds)
        # print(dct_embeds.size())
        batch_size, seq_len, _ = dct_embeds.size()

        pos_embeds = self.position_embeddings(pos_in.squeeze(1))
        pos_embeds = self.embedding_dropout(pos_embeds)
        # print(pos_embeds.view(batch_size, max_len, -1).size())

        dct_out = self.dct_detector(dct_embeds, pos_embeds.view(batch_size, seq_len, -1))
        return dct_out


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





