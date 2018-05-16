
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

torch.manual_seed(2)


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

dct_inputs = torch.LongTensor(500, 1, 3).random_(0, 100)
time_inputs = torch.LongTensor(500, 1, 2, 3).random_(0, 100)

targets = torch.Tensor(500, 3).random_(0, 2)

print(targets)

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
                             [1, 1, 1]], dtype=torch.int16)


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
            
        
# class diff_elem_loss(nn.Module):
    
#     def __init__(self):
#         super(diff_elem_loss, self).__init__()
    
#     def forward(self, pred_out, target):
#         diff_t = torch.eq(pred_out, target)
#         sum_elem = pred_out.numel()
#         loss = ( sum_elem - diff_t.sum().item() ) / sum_elem
#         return autograd.Variable(torch.FloatTensor([loss]), requires_grad=True)

    
# def diff_elem_loss(pred_out, target)
#     diff_t = torch.eq(pred_out, target)
#     sum_elem = pred_out.nelement()
#     loss = ( sum_elem - diff_t.sum().item() ) / sum_elem
#     return autograd.Variable(torch.FloatTensor([loss]), requires_grad=True)
    
class distance_loss(nn.Module):
    
    def __init__(self):
        super(distance_loss, self).__init__()
    
    def forward(self, action_out, target):
        pass


# In[43]:


class DCTDetector(nn.Module):
    def __init__(self, word_embeddings, embedding_dim, dct_hidden_dim, action_size, batch_size):
        super(DCTDetector, self).__init__()
        ## initialize parameters
        self.dct_hidden_dim = dct_hidden_dim
        self.batch_size = batch_size
        
        ## initialize neural layers
        self.word_embeddings = word_embeddings
        self.embedding_dropout = nn.Dropout(p=0.5)
        self.dct_tagger = nn.LSTM(embedding_dim, dct_hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.tagger_dropout = nn.Dropout(p=0.5)
        self.dct_hidden2action = nn.Linear(dct_hidden_dim, action_size)
        self.dct_hidden = self.init_dct_hidden()
        
    def init_dct_hidden(self):
        return (torch.zeros(2, self.batch_size, self.dct_hidden_dim // 2),
                torch.zeros(2, self.batch_size, self.dct_hidden_dim // 2))
          
    def forward(self, dct_input):
        # print(dct_input)
        dct_var = dct_input.view(self.batch_size, -1)
        dct_embeds = self.word_embeddings(dct_var).view(self.batch_size, dct_input.size()[-1], -1)
        dct_embeds = self.embedding_dropout(dct_embeds)
        # print(dct_embeds)
        dct_out, self.dct_hidden = self.dct_tagger(dct_embeds, self.dct_hidden)
        dct_out = self.dct_hidden2action(dct_out[:, -1, :])
        # print(dct_out)
        dct_score = F.log_softmax(dct_out, dim=1)
        # print(dct_score)
        return dct_score
        
class TimeDetector(nn.Module):
    
    def __init__(self, word_embeddings, embedding_dim, time_hidden_dim, action_size, batch_size):
        super(TimeDetector, self).__init__()
        ## initialize parameters
        self.time_hidden_dim = time_hidden_dim
        self.batch_size = batch_size
        
        
        ## initialize neural layers
        self.word_embeddings = word_embeddings
        self.left_embedding_dropout = nn.Dropout(p=0.5)
        self.right_embedding_dropout = nn.Dropout(p=0.5)
        self.left_tagger = nn.LSTM(embedding_dim, time_hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.left_tagger_dropout = nn.Dropout(p=0.5)
        self.right_tagger = nn.LSTM(embedding_dim, time_hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.right_tagger_dropout = nn.Dropout(p=0.5)
        self.time_hidden2action = nn.Linear(time_hidden_dim * 2, action_size)
        self.left_hidden = self.init_left_time_hidden()
        self.right_hidden = self.init_right_time_hidden()
        
    def init_left_time_hidden(self):
        return (torch.zeros(2, self.batch_size, self.time_hidden_dim // 2),
                torch.zeros(2, self.batch_size, self.time_hidden_dim // 2))
    
    def init_right_time_hidden(self):
        return (torch.zeros(2, self.batch_size, self.time_hidden_dim // 2),
                torch.zeros(2, self.batch_size, self.time_hidden_dim // 2))
    
    def forward(self, left_input, right_input):
        
        ## left branch
        left_var = left_input
        left_embeds = self.word_embeddings(left_var).view(self.batch_size, left_input.size()[-1], -1)
        left_embeds = self.left_embedding_dropout(left_embeds)
        left_out, self.left_hidden = self.left_tagger(left_embeds, self.left_hidden)
        left_out = self.left_tagger_dropout(left_out[:,-1, :])

        ## right branch
        right_var = right_input
        right_embeds = self.word_embeddings(right_var).view(self.batch_size, right_input.size()[-1], -1)
        right_embeds = self.right_embedding_dropout(right_embeds)
        right_out, self.right_hidden = self.right_tagger(right_embeds, self.right_hidden)
        right_out = self.right_tagger_dropout(right_out[:,-1, :])

        ## concatenation

        time_out = torch.cat((left_out, right_out), 1)
        time_out = self.time_hidden2action(time_out)
        time_score = F.log_softmax(time_out, dim=1)
        return time_score
                                       
class TimeInferrer(nn.Module):
                                       
    def __init__(self, embedding_dim, dct_hidden_dim, time_hidden_dim, vocab_size, action_size, batch_size):
        super(TimeInferrer, self).__init__()
        self.dct_hidden_dim = dct_hidden_dim
        self.time_hidden_dim = time_hidden_dim
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim                              
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dct_detector = DCTDetector(self.word_embeddings, 
                                        self.embedding_dim, 
                                        self.dct_hidden_dim, 
                                        action_size, 
                                        batch_size)
        self.time_detector = TimeDetector(self.word_embeddings, 
                                          self.embedding_dim, 
                                          self.time_hidden_dim, 
                                          action_size, 
                                          batch_size)
                                       
    def forward(self, dct_input, time_inputs):
        
        ## event to dct
        dct_score = self.dct_detector(dct_input)
        out_scores = dct_score.clone().view(self.batch_size, 1, -1)
#         print(out_scores.size())
        
        ## event to per time expression
        for time_index in range(time_inputs.size()[1]):
            left_input = time_inputs[:, time_index, 0, :]
            right_input = time_inputs[:, time_index, 1, :]
            time_score = self.time_detector(left_input, right_input).view(self.batch_size, 1, -1)
#             print(time_score.size())
            out_scores = torch.cat((out_scores, time_score), dim=1)
#             print(out_scores.size())
            
        
        return out_scores


# In[98]:




def score2pred_2(out_scores, norm_times, BATCH_SIZE, update_strategies):
    norm_times = torch.tensor([[0, 0, 0], [1, 1, 1]])
    preds_out = torch.ones(BATCH_SIZE, 3) * -1
    preds_out.requires_grad_()
    class_ids = torch.argmax(out_scores, dim=2)
    for i in range(out_scores.size()[0]):
        for cid, time in zip(class_ids[i], norm_times):
            preds_out[i] = update_strategies[cid].float() * time.float() + (torch.ones_like(update_strategies[cid]) - update_strategies[cid]).float() * preds_out[i].float()
    return preds_out

def diff_elem_loss(pred_out, target):
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
        if not param.requires_grad:
            print(name)
        else:
            print('*', name)
    ## train model
    a = list(model.parameters())[1]

    for epoch in range(EPOCH_NUM):

        total_loss = torch.Tensor([0])
        for step, (dct_input, time_input, target) in enumerate(loader):


            model.zero_grad()

            time_scores = model(dct_input, time_input)
    #         print(time_scores.requires_grad)
            pred_out = score2pred_2(time_scores, None, BATCH_SIZE, update_strategies)
    #         print(pred_out.requires_grad)
            loss = diff_elem_loss(pred_out, target)

    #         print(loss)
    #         print(list(model.parameters())[0].grad)
    #         print(loss)
    #         print(loss.grad)
            loss.backward(retain_graph=True)
            optimizer.step()
            ## check if the model parameters being updated

    #         print(len(model.word_embeddings.weight[1]))
    #         print(list(model.parameters())[0].grad)


            total_loss += loss.data.item() * pred_out.size()[0]
    #         print('')
    #     for name, param in model.named_parameters():
    #         if name == 'time_detector.time_hidden2action.weight':
    #             print(param.grad)
        print(model.word_embeddings(torch.LongTensor([[0, ]])).squeeze()[:5])
        b = list(model.parameters())[1]
        print('Epoch', epoch, 'Step', step, ', Epoch loss: %.4f' % (total_loss / 500), torch.equal(a.data, b.data))
    #     print(model.word_embeddings.data)
        
if __name__ == "__main__":
    main()


# In[ ]:


class TimeInferrer(nn.Module):
    
    def __init__(self, embedding_dim, dct_hidden_dim, time_hidden_dim, vocab_size, action_size, batch_size):
        super(TimeInferrer, self).__init__()
        self.dct_hidden_dim = dct_hidden_dim
        self.time_hidden_dim = time_hidden_dim
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        ## DCT and timex relation tagger
        self.dct_tagger = nn.LSTM(embedding_dim, dct_hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.left_time_tagger = nn.LSTM(embedding_dim, time_hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.right_time_tagger = nn.LSTM(embedding_dim, time_hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        
        ## The linear layer that maps from LSTM output to action space
        self.dct_hidden2action = nn.Linear(dct_hidden_dim, action_size)
        self.time_hidden2action = nn.Linear(time_hidden_dim * 2, action_size)
        
        self.dct_hidden = self.init_dct_hidden()
#         self.left_time_hidden = self.init_left_time_hidden()
#         self.right_time_hidden = self.init_right_time_hidden()
        
    def init_dct_hidden(self):
        return (autograd.Variable(torch.zeros(2, self.batch_size, self.dct_hidden_dim // 2)),
                autograd.Variable(torch.zeros(2, self.batch_size, self.dct_hidden_dim // 2)))
    
    def init_left_time_hidden(self):
        return (autograd.Variable(torch.zeros(2, self.batch_size, self.time_hidden_dim // 2)),
                autograd.Variable(torch.zeros(2, self.batch_size, self.time_hidden_dim // 2)))
    
    def init_right_time_hidden(self):
        return (autograd.Variable(torch.zeros(2, self.batch_size, self.time_hidden_dim // 2)),
                autograd.Variable(torch.zeros(2, self.batch_size, self.time_hidden_dim // 2)))
    
    def forward(self, dct_in, time_inputs):
        
#         print(dct_in.size(), time_inputs.size())
        ## store all the dct_score and time_score into one list for calculating loss
#         print(self.word_embeddings.weight.data[48])
        
        ## event to dct
        dct_var = autograd.Variable(dct_in.view(self.batch_size, -1))
        dct_embeds = self.word_embeddings(dct_var).view(self.batch_size, dct_in.size()[-1], -1)
        dct_out, self.dct_hidden = self.dct_tagger(dct_embeds, self.dct_hidden)
        dct_score = self.dct_hidden2action(dct_out[:, -1, :])
        time_scores = dct_score.clone().view(self.batch_size, 1, -1)
        
        ## event to per time expression
        for time_index in range(time_inputs.size()[1]):
            
            
            self.left_time_hidden = self.init_left_time_hidden()
            self.right_time_hidden = self.init_right_time_hidden()
            
            ## left branch
            left_time = time_inputs[:, time_index, 0, :]
            left_time_var = autograd.Variable(left_time)
            left_time_embeds = self.word_embeddings(left_time_var).view(self.batch_size, left_time.size()[-1], -1)
            left_time_out, self.left_time_hidden = self.left_time_tagger(left_time_embeds, self.left_time_hidden)
            
            ## right branch
            right_time = time_inputs[:, time_index, 1, :]
            right_time_var = autograd.Variable(right_time)
            right_time_embeds = self.word_embeddings(right_time_var).view(self.batch_size, right_time.size()[-1], -1)
            right_time_out, self.right_time_hidden = self.right_time_tagger(right_time_embeds, self.right_time_hidden)
            
            ## concatenation
            
            time_out = torch.cat((left_time_out[:,-1, :], right_time_out[:,-1, :]), 1)
            time_score = self.time_hidden2action(time_out)
            time_scores = torch.cat((time_scores, time_score.view(self.batch_size, 1, -1)), 1)
        
        
        return time_scores

