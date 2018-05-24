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
    print("[Pre-trained embeddings] weight size: ", pre_weights.size())
    return nn.Embedding.from_pretrained(pre_weights, freeze=True)





def geneInterPostion(word_list):
    position_list = []
    word_len = len(word_list)
    for i in range(word_len):
        position_list.append((i - 0, i - word_len + 1))
    return position_list


