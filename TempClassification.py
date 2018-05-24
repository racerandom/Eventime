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
import TempData

torch.manual_seed(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)



word_in, pos_in, rel_in, max_len, pos_idx, word_idx, rel_idx = TempData.prepare()

print(word_in.size(), pos_in.size(), rel_in.unsqueeze(1).size())
print(word_in.dtype, pos_in.dtype, rel_in.unsqueeze(1).dtype)



VOCAB_SIZE = len(word_idx)
POS_SIZE = len(pos_idx)
MAX_LEN = max_len
ACTION_SIZE = len(rel_idx)
BATCH_SIZE = 20



dataset = Tlink.MultipleDatasets(word_in, pos_in, rel_in)

loader = Data.DataLoader(
                        dataset=dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=1,
                        )

EMBEDDING_DIM = 200
POSITION_DIM = 20
DCT_HIDDEN_DIM = 100
TIME_HIDDEN_DIM = 50
FC1_DIM = 100
WINDOW = 3
EPOCH_NUM = 50
learning_rate = 0.01


class TempClassifier(nn.Module):
    def __init__(self, embedding_dim, position_dim, hidden_dim, vocab_size, pos_size, seq_len, fc1_dim, action_size,
                 batch_size, window, pre_model=None):
        super(TempClassifier, self).__init__()
        self.batch_size = batch_size
        self.word_embeddings = TempUtils.pre2embed(pre_model) if pre_model else nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(pos_size, position_dim)
        self.embedding_dropout = nn.Dropout(p=0.5)
        # self.dct_detector = Tlink.DCTDetector(embedding_dim,
        #                                       dct_hidden_dim,
        #                                       action_size,
        #                                       batch_size)
        self.dct_detector = Tlink.TempCNN(embedding_dim, position_dim, hidden_dim, seq_len, fc1_dim, action_size, window)

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


# pre_model, word2ix = TempUtils.load_pre()
model = TempClassifier(EMBEDDING_DIM, POSITION_DIM, DCT_HIDDEN_DIM, VOCAB_SIZE, POS_SIZE, MAX_LEN, FC1_DIM, ACTION_SIZE,
                       BATCH_SIZE, WINDOW, pre_model=None).to(device=device)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate) ##  fixed a error when using pre-trained embeddings
print(model)
for name, param in model.named_parameters():
    if param.requires_grad:
        print('*', name)

param_id = 3
for epoch in range(EPOCH_NUM):
    total_loss = []
    a = list(model.parameters())[param_id].clone()
    start_time = time.time()
    for step, (dct_input, position_input, target) in enumerate(loader):
        dct_input, position_input, target = dct_input.to(device), position_input.to(device), target.to(device)

        model.zero_grad()
        dct_out = model(dct_input, position_input)
        loss = loss_function(dct_out, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        total_loss.append(loss.data.item())
        # print('epoch %.4f' % loss.item(), '%.5s seconds' % (time.time() - start_time))
    # for name, param in model.named_parameters():
    #     if param.requires_grad and name == 'dct_detector.c1.weight':
    #         print(param[0][:5])
    # print(model.word_embeddings(torch.LongTensor([[0, ]])).squeeze()[:5])
        b = list(model.parameters())[param_id].clone()

        # print(torch.equal(a.data, b.data))
    print('Epoch %i' % epoch, ',loss: %.4f' % (sum(total_loss) / float(len(total_loss))), ', %.5s seconds' % (time.time() - start_time))
