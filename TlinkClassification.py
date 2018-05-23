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
import time

import TempModules as Tlink

torch.manual_seed(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dct_inputs = torch.randint(0, 100, (500, 1, 10), dtype=torch.long, device=device)
time_inputs = torch.randint(0, 100, (500, 1, 2, 3), dtype=torch.long, device=device)

targets = torch.randint(0, 2, (500, 1), dtype=torch.long, device=device)

# dct_inputs = torch.LongTensor(500, 1, 3).random_(0, 100)
# time_inputs = torch.LongTensor(500, 1, 2, 3).random_(0, 100)
#
# targets = torch.Tensor(500, 3).random_(0, 2)

BATCH_SIZE = 100
# EPOCH_NUM = 100

print(dct_inputs.size(), time_inputs.size(), targets.size())

dataset = Tlink.MultipleDatasets(dct_inputs, time_inputs, targets)

loader = Data.DataLoader(
    dataset = dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 1,
)

EMBEDDING_DIM = 64
DCT_HIDDEN_DIM = 60
TIME_HIDDEN_DIM = 50
FC1_DIM = 30
VOCAB_SIZE = 100
ACTION_SIZE = 2
EPOCH_NUM = 50
learning_rate = 0.01

class DCTClassifier(nn.Module):
    def __init__(self, embedding_dim, dct_hidden_dim, vocab_size, fc1_dim, action_size, batch_size):
        super(DCTClassifier, self).__init__()
        self.dct_hidden_dim = dct_hidden_dim
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(p=0.5)
        # self.dct_detector = Tlink.DCTDetector(self.embedding_dim,
        #                                       self.dct_hidden_dim,
        #                                       action_size,
        #                                       batch_size)
        self.dct_detector = Tlink.DCTCNN(self.embedding_dim, self.dct_hidden_dim, 10, fc1_dim, action_size, 3)
    def forward(self, dct_in):
        dct_var = dct_input.view(self.batch_size, -1)
        dct_embeds = self.word_embeddings(dct_var).view(self.batch_size, dct_input.size()[-1], -1)
        dct_embeds = self.embedding_dropout(dct_embeds)
        dct_out = self.dct_detector(dct_embeds)
        return dct_out



model = DCTClassifier(EMBEDDING_DIM, DCT_HIDDEN_DIM, VOCAB_SIZE, FC1_DIM, ACTION_SIZE, BATCH_SIZE).to(device=device)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(model)
for name, param in model.named_parameters():
    if param.requires_grad:
        print('*', name)
param_id = 3
for epoch in range(EPOCH_NUM):
    total_loss = torch.Tensor([0])
    a = list(model.parameters())[param_id].clone()
    for step, (dct_input, time_input, target) in enumerate(loader):
        start_time = time.time()
        model.zero_grad()
        dct_out = model(dct_input)
        loss = loss_function(dct_out, torch.squeeze(target, 1))
        loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += loss.data.item() * dct_out.size()[0]
        # print('epoch %.4f' % loss.item(), '%.5s seconds' % (time.time() - start_time))
    # for name, param in model.named_parameters():
    #     if param.requires_grad and name == 'dct_detector.c1.weight':
    #         print(param[0][:5])
    # print(model.word_embeddings(torch.LongTensor([[0, ]])).squeeze()[:5])
        b = list(model.parameters())[param_id].clone()

        # print(torch.equal(a.data, b.data))
    print('Epoch loss: %.4f' % (total_loss.item()), ', %.5s seconds' % (time.time() - start_time))
