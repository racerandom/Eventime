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
import TempUtils

torch.manual_seed(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

VOCAB_SIZE = 100
POS_SIZE = 10
MAX_LEN = 10
ACTION_SIZE = 2

dct_inputs = torch.randint(0, VOCAB_SIZE, (500, 1, MAX_LEN), dtype=torch.long)
position_inputs = torch.randint(0, POS_SIZE, (500, MAX_LEN, 2), dtype=torch.long)
time_inputs = torch.randint(0, VOCAB_SIZE, (500, 1, 2, 3), dtype=torch.long)

targets = torch.randint(0, ACTION_SIZE, (500, 1), dtype=torch.long)

# dct_inputs = torch.LongTensor(500, 1, 3).random_(0, 100)
# time_inputs = torch.LongTensor(500, 1, 2, 3).random_(0, 100)
#
# targets = torch.Tensor(500, 3).random_(0, 2)

BATCH_SIZE = 100

print(dct_inputs.size(), time_inputs.size(), targets.size())

dataset = Tlink.MultipleDatasets(dct_inputs, position_inputs, time_inputs, targets)

loader = Data.DataLoader(
                        dataset=dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=1,
                        )

EMBEDDING_DIM = 200
POSITION_DIM = 20
DCT_HIDDEN_DIM = 60
TIME_HIDDEN_DIM = 50
FC1_DIM = 30
WINDOW = 3
EPOCH_NUM = 50
learning_rate = 0.01


class TempClassifier(nn.Module):
    def __init__(self, pre_model, embedding_dim, position_dim, hidden_dim, vocab_size, pos_size, max_len, fc1_dim, action_size, batch_size, window):
        super(TempClassifier, self).__init__()
        self.batch_size = batch_size
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pre2embed(pre_model)
        self.position_embeddings = nn.Embedding(10, position_dim)
        self.embedding_dropout = nn.Dropout(p=0.5)
        # self.dct_detector = Tlink.DCTDetector(embedding_dim,
        #                                       dct_hidden_dim,
        #                                       action_size,
        #                                       batch_size)
        self.dct_detector = Tlink.TempCNN(embedding_dim, position_dim, hidden_dim, max_len, fc1_dim, action_size, window)

    def pre2embed(self, pre_model):
        pre_weights = torch.FloatTensor(pre_model.vectors)
        print("[Pre-trained embeddings] weight size: ",pre_weights.size())
        self.word_embeddings = nn.Embedding.from_pretrained(pre_weights, freeze=True)
        self.word_embeddings.weight.requires_grad = False

    def forward(self, dct_in, pos_in):
        # print(dct_input.size(), pos_in.size())

        dct_embeds = self.word_embeddings(dct_input.squeeze(1))
        dct_embeds = self.embedding_dropout(dct_embeds)
        # print(dct_embeds.size())
        batch_size, seq_len, _ = dct_embeds.size()

        pos_embeds = self.position_embeddings(pos_in.squeeze(1))
        pos_embeds = self.embedding_dropout(pos_embeds)
        # print(pos_embeds.view(batch_size, max_len, -1).size())

        dct_out = self.dct_detector(dct_embeds, pos_embeds.view(batch_size, seq_len, -1))
        return dct_out


pre_model, word2ix = TempUtils.load_pre()
model = TempClassifier(pre_model, EMBEDDING_DIM, POSITION_DIM, DCT_HIDDEN_DIM, VOCAB_SIZE, POS_SIZE, MAX_LEN, FC1_DIM, ACTION_SIZE, BATCH_SIZE, WINDOW).to(device=device)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
print(model)
for name, param in model.named_parameters():
    if param.requires_grad:
        print('*', name)

param_id = 3
for epoch in range(EPOCH_NUM):
    total_loss = torch.Tensor([0])
    a = list(model.parameters())[param_id].clone()
    for step, (dct_input, position_input, time_input, target) in enumerate(loader):
        dct_input, position_input, time_input, target = dct_input.to(device), position_input.to(device), time_input.to(device), target.to(device)
        start_time = time.time()
        model.zero_grad()
        dct_out = model(dct_input, position_input)
        loss = loss_function(dct_out, target.squeeze(1))
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
    print('Epoch %i' % epoch, ',loss: %.4f' % (total_loss.item()), ', %.5s seconds' % (time.time() - start_time))
