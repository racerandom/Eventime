import torch
import torch.nn as nn
import gensim
import numpy as np

import time


def load_pre(embed_file, binary=True, addZeroPad=True):
    word2ix = {}
    start_time = time.time()
    pre_model = gensim.models.KeyedVectors.load_word2vec_format(embed_file, binary=binary)
    for word, value in pre_model.vocab.items():
        word2ix[word] = value.index + 1 if addZeroPad else 0
    pre_vectors = pre_model.vectors
    embed_size = pre_vectors.shape[-1]
    print("[Pre-trained embeddings] file='%s' loaded, took %.5s seconds...]" % (embed_file, time.time() - start_time))
    if addZeroPad:
        pre_vectors = np.concatenate((np.zeros((1, embed_size)), pre_vectors), axis=0)
    return pre_vectors, word2ix


def embed_txt2bin(embed_file):
    pass


def pre2embed(pre_vectors):
    pre_weights = torch.FloatTensor(pre_vectors)
    #print("[Pre-trained embeddings] weight size: ", pre_weights.size())
    return nn.Embedding.from_pretrained(pre_weights, freeze=True)


def tok2ix(tok, to_ix, unk_ix=0):
    return to_ix[tok] if tok in to_ix else unk_ix


## convert 1D token sequences to token_index sequences
def prepare_seq_1d(seq_1d, to_ix, unk_ix=0):
    ix_seq_1d = [tok2ix(tok, to_ix, unk_ix=unk_ix) for tok in seq_1d]
    return ix_seq_1d


## convert 2D token sequences to token_index sequences
def prepare_seq_2d(seq_2d, to_ix, unk_ix=0):
    ix_seq_2d = [[tok2ix(tok, to_ix, unk_ix=unk_ix) for tok in seq_1d] for seq_1d in seq_2d]

    return ix_seq_2d


def prepare_sequence_pos(seq_2d, to_ix):
    ix_seq = [[[to_ix[p_l], to_ix[p_r]] for p_l, p_r in seq_1d] for seq_1d in seq_2d]
    return ix_seq


## padding 2D index sequences to a fixed given length
def padding_2d(seq_2d, max_len, pad=0, direct='right'):

    for seq_1d in seq_2d:
        for i in range(0, max_len - len(seq_1d)):
            if direct in ['right']:
                seq_1d.append(pad)
            else:
                seq_1d.insert(0, pad)
    return seq_2d


def padding_pos(seq_2d, max_len, pad=[0, 0], direct='right'):

    for seq_1d in seq_2d:
        for i in range(0, max_len - len(seq_1d)):
            if direct in ['right']:
                seq_1d.append(pad)
            else:
                seq_1d.insert(0, pad)
    return seq_2d


def pos2idx(doc_dic, link_type):
    tok_idx = {'zeropadding': 0}
    for doc in doc_dic.values():
        for link in doc.get_links_by_type(link_type):
            for tok_l, tok_r in link.interpos:
                tok_idx.setdefault(tok_l, len(tok_idx))
                tok_idx.setdefault(tok_r, len(tok_idx))
    return tok_idx


def feat2idx(doc_dic, feat_name, link_type, feat_idx=None):
    for doc in doc_dic.values():
        for link in doc.get_links_by_type(link_type):
            for feat in link.feat_inputs[feat_name]:
                feat_idx.setdefault(feat, len(feat_idx))
    return feat_idx


def word2idx(doc_dic, link_type):
    tok_idx = {'zeropadding': 0}
    for doc in doc_dic.values():
        for link in doc.get_links_by_type(link_type):
            for tok in link.interwords:
                tok_idx.setdefault(tok, len(tok_idx))
    return tok_idx


def rel2idx(doc_dic, link_type):
    tok_idx = {}
    for doc in doc_dic.values():
        for link in doc.get_links_by_type(link_type):
            tok_idx.setdefault(link.rel, len(tok_idx))
    return tok_idx


def max_length(doc_dic, feat_name, link_type):
    word_list = []
    for doc in doc_dic.values():
        for link in doc.get_links_by_type(link_type):
            word_list.append(len(link.feat_inputs[feat_name]))
    return max(word_list)


def geneInterPostion(word_list):
    position_list = []
    word_len = len(word_list)
    for i in range(word_len):
        position_list.append((i - 0, i - word_len + 1))
    return position_list


def geneSentPostion(tokens, sour, targ):
    left_pos, right_pos = [], []
    for tok in tokens:
        left_pos.append(tok.tok_id - sour.tok_ids[-1])
        right_pos.append(tok.tok_id - targ.tok_ids[-1])
    return left_pos, right_pos


def getEndPosition(sour, targ):
    left_mention, right_mention = [], []
    for tok_id in sour.tok_ids:
            left_mention.append(tok_id - targ.tok_ids[-1])
    for tok_id in targ.tok_ids:
            right_mention.append(tok_id - sour.tok_ids[-1])
    return left_mention, right_mention

def getMentionDist(tokens, sour):
    dist = []
    for tok in tokens:
        dist.append(tok.tok_id - sour.tok_ids[-1])
    return dist


def dict2str(dic):
    out = []
    for k, v in dic.items():
        out.append('%s=%s' % (k, v))
    return ','.join(out)


def save_checkpoint(state, is_best, filename):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        torch.save(state, filename)  # save checkpoint
        return "=> Saving a new best"
    else:
        return ""