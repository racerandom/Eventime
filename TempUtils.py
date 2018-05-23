import torch
import torch.nn as nn
import gensim

import time


def load_pre(file_name='/Users/fei-c/Resources/embed/giga-aacw.d200.bin', binary=True):
    word2ix = {}
    start_time = time.time()
    pre_model = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=binary)
    for word, value in pre_model.vocab.items():
        word2ix[word] = value.index
    print("[Pre-trained embeddings] file='%s' loaded, took %.5s seconds...]" % (file_name, time.time() - start_time))
    return pre_model, word2ix


def pre2embed(pre_model):
    pre_weights = torch.FloatTensor(pre_model.vectors)
    print(pre_weights.size())
    pre_embed = nn.Embedding.from_pretrained(pre_weights, freeze=True)
    return pre_embed


def padding(inputs, pad=0, direct='right'):
    max_len = max([len(x) for x in inputs])
    for x in inputs:
        for i in range(0, max_len - len(x)):
            if direct in ['right']:
                x.append(pad)
            else:
                x.insert(0, pad)
    return torch.LongTensor(inputs)

