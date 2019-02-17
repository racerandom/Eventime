import os
import torch
import torch.nn as nn
import gensim
import numpy as np
import pickle
import logging
import time


def setup_stream_logger(logger_name, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)

    return logger


def load_pre(embed_file, binary=True, addZeroPad=True):
    word2ix = {}
    start_time = time.time()
    pre_model = gensim.models.KeyedVectors.load_word2vec_format(embed_file, binary=binary)
    if addZeroPad:
        word2ix['zeropadding'] = 0
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


def pre2embed(pre_vectors, freeze_mode):
    pre_weights = torch.FloatTensor(pre_vectors)
    return nn.Embedding.from_pretrained(pre_weights, freeze=freeze_mode)


def tok2ix(tok, to_ix, unk_ix=0):
    return to_ix[tok] if tok in to_ix else unk_ix


## convert 1D token sequences to token_index sequences
def prepare_seq_1d(seq_1d, to_ix, unk_ix=0):
    ix_seq_1d = [tok2ix(tok, to_ix, unk_ix=unk_ix) for tok in seq_1d]
    return ix_seq_1d

def startIndexOfLastSent(sent_seq):
    lastSentId = sent_seq[-1].sent_id
    for index, tok in enumerate(sent_seq):
        if tok.sent_id == lastSentId:
            return index

def reviceSdpWithSentID(sent_seq, sdp_conll_ids):
    startOfLastSent = startIndexOfLastSent(sent_seq)
    return [ startOfLastSent + conll_id for conll_id in sdp_conll_ids]


## convert 2D token sequences to token_index sequences
def prepare_seq_2d(seq_2d, to_ix, unk_ix=0):
    ix_seq_2d = [[tok2ix(tok, to_ix, unk_ix=unk_ix) for tok in seq_1d] for seq_1d in seq_2d]

    return ix_seq_2d


## convert 3D char sequences to char_index sequences
def prepare_seq_3d(seq_3d, to_ix, unk_ix=0):
    ix_seq_3d = [[[tok2ix(char, to_ix, unk_ix=unk_ix) for char in seq_1d] for seq_1d in seq_2d] for seq_2d in seq_3d]

    return ix_seq_3d


def prepare_sequence_pos(seq_2d, to_ix):
    ix_seq = [[[to_ix[p_l], to_ix[p_r]] for p_l, p_r in seq_1d] for seq_1d in seq_2d]
    return ix_seq


## padding 2D index sequences to a fixed given length
def padding_2d(seq_2d, max_seq_len, pad=0, direct='right'):

    for seq_1d in seq_2d:
        for i in range(0, max_seq_len - len(seq_1d)):
            if direct in ['right']:
                seq_1d.append(pad)
            else:
                seq_1d.insert(0, pad)
    return seq_2d


## padding 3D char index sequences with fixed max_seq_len, max_char_len
def padding_3d(seq_3d, max_seq_len, max_char_len, pad=0, direct='right'):

    for seq_2d in seq_3d:

        for seq_1d in seq_2d:
            for i in range(max_char_len - len(seq_1d)):
                if direct in ['right']:
                    seq_1d.append(pad)
                else:
                    seq_1d.insert(0, pad)

        for j in range(max_seq_len - len(seq_2d)):
            if direct in ['right']:
                seq_2d.append([pad] * max_char_len)
            else:
                seq_2d.insert(0, [pad] * max_char_len)
    return seq_3d


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


def label_to_ix(targs):
    targ2ix = {}
    for targ in targs:
        targ2ix.setdefault(targ, len(targ2ix))
    return targ2ix


def wvocab2cvocab(word_vocab):
    char_vocab = set()
    for word in word_vocab:
        char_vocab.update(list(word.lower()))
    return char_vocab


def doc2fvocab(doc_dic, feat_name, link_types):
    feat_vocab = set()
    for doc in doc_dic.values():
        for link_type in link_types:
            for link in doc.get_links_by_type(link_type):
                if feat_name in link.feat_inputs:
                    for feat in link.feat_inputs[feat_name]:
                        feat_vocab.add(feat)
    return feat_vocab

def doc2fvocab2(doc_dic, task, feat_names):
    feat_list = []
    for feat_name in feat_names:
        feats = doc2feat(doc_dic, feat_name, task)
        if feats:
            feat_list.extend(feats)
    return feat2vocab(feat_list)

def feat2vocab(feat_list):
    if feat_list:
        vocab = set()
        for line in feat_list:
            for tok in line:
                vocab.add(tok)
        return vocab
    else:
        return None


def doc2feat(doc_dic, feat_name, task):
    feat_list = []
    for doc in doc_dic.values():
        if task in ['day_len']:
            for event in doc.events.values():
                if feat_name in event.feat_inputs:
                    feat_list.append(event.feat_inputs[feat_name])
                else:
                    return None
        elif task in ['Event-Event', 'Event-Timex', 'Event-DCT']:
            for tlink in doc.get_links_by_type(task):
                if feat_name in tlink.feats_inputs:
                    feat_list.append(event.feat_inputs[feat_name])
                else:
                    return None
    return feat_list


def doc2wvocab(doc_dic):
    wvocab = set()
    for doc in doc_dic.values():
        for tok in doc.tokens:
            wvocab.add(tok.content)
    return wvocab


def doc2featList(doc_dic, dataset, feat_name, link_types):
    featList = []
    for doc_id, doc in doc_dic.items():
        if doc_id not in dataset:
            continue
        for link_type in link_types:
            for link in doc.get_links_by_type(link_type):
                if feat_name in link.feat_inputs:
                    feat.append(link.feat_inputs[feat_name])
    return featList


def vocab2idx(vocab, feat_idx=None):
    if vocab:
        if not feat_idx:
            feat_idx = {}
        for feat in vocab:
            feat_idx.setdefault(feat, len(feat_idx))
        return feat_idx
    else:
        return None


def feat_to_ix(feats, feat2ix=None):
    if feat2ix is None:
        feat2ix = {'zeropadding': 0}
    for feat_sample in feats:
        for tok in feat_sample:
            feat2ix.setdefault(tok, len(feat2ix))
    return feat2ix


def word2idx(doc_dic, link_types):
    tok_idx = {'zeropadding': 0}
    for doc in doc_dic.values():
        for link_type in link_types:
            for link in doc.get_links_by_type(link_type):
                for tok in link.interwords:
                    tok_idx.setdefault(tok, len(tok_idx))
    return tok_idx


def pickle_data(data, pickle_file='data/temp.pkl'):
    with open(pickle_file, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(data, f)
    print("Successfully save '%s'..." % pickle_file)


def load_pickle(pickle_file='data/temp.pkl'):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    print("Successfully load data from pickle file '%s'..." % pickle_file)
    return data


def pre2embed(pre_vectors, freeze_mode):
    pre_weights = torch.FloatTensor(pre_vectors)
    return nn.Embedding.from_pretrained(pre_weights, freeze=freeze_mode)


def slim_word_embed(word2ix, embed_file, embed_pickle_file):



    def load_pre_embed(embed_file, binary):
        if embed_file and os.path.isfile(os.path.join(os.getenv("HOME"), embed_file)):
            start_time = time.time()
            pre_embed = gensim.models.KeyedVectors.load_word2vec_format(embed_file, binary=binary)
            pre_word2ix = {}
            for word, value in pre_embed.vocab.items():
                pre_word2ix[word] = value.index
            print("[Embedding] Successfully load the pre-trained embedding file '%s' in %i seconds..." % (embed_file,
                                                                                                          time.time() - start_time))
            return pre_word2ix, pre_embed.vectors
        else:
            raise Exception("[ERROR]Cannot find the pre-trained embedding file...")

    def pre_embed_to_weight(word2ix, embed_file, binary=True):
        pre_word2ix, pre_weights = load_pre_embed(embed_file, binary)
        count = 0
        word_dim = pre_weights.shape[1]
        weights = []
        for key, ix in sorted(word2ix.items(), key=lambda x: x[1]):
            if ix == 0:
                weights.append(np.zeros(word_dim))
            else:
                if key in pre_word2ix:
                    count += 1
                    weights.append(pre_weights[pre_word2ix[key]])
                elif key.lower() in pre_word2ix:
                    count += 1
                    weights.append(pre_weights[pre_word2ix[key.lower()]])
                else:
                    weights.append(np.random.uniform(-1, 1, word_dim))
        slim_weights = np.stack(weights, axis=0)
        print("[Embedding] Successfully the slim embedding weights from %i to %i words, "
              "%i words (%.2f%%) are covered" % (len(pre_word2ix),
                                                 len(word2ix),
                                                 count,
                                                 100 * count / len(word2ix)))
        return slim_weights

    embed_weights = pre_embed_to_weight(word2ix, embed_file)
    pickle_data((word2ix, embed_weights), pickle_file=embed_pickle_file)


def rel2idx(doc_dic, task):
    tok_idx = {}
    for doc in doc_dic.values():
        if task in ['day_len']:
            for event in doc.events.values():
                tok_idx.setdefault(event.daylen,len(tok_idx))
        elif task in ['Event-Event', 'Event-Timex', 'Event-DCT']:
            for link in doc.get_links_by_type(task):
                tok_idx.setdefault(link.rel, len(tok_idx))
    return tok_idx


def sizeOfVocab(vocab):
    if vocab:
        return len(vocab)
    else:
        return 0


def max_length(doc_dic, task, feat_names):
    feat_list = []
    for feat_name in feat_names:
        feats = doc2feat(doc_dic, feat_name, task)
        if feats:
            feat_list.extend(feats)
    if feat_list:
        return max([len(feat) for feat in feat_list])
    else:
        return 0


def max_len_2d(seq_2d):
    return max([len(seq_1d) for seq_1d in seq_2d])


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


# def setup_logger(logger_name, log_file, level=logging.INFO):
#     l = logging.getLogger(logger_name)
#     formatter = logging.Formatter('%(message)s')
#     fileHandler = logging.FileHandler(log_file, mode='w')
#     fileHandler.setFormatter(formatter)
#     streamHandler = logging.StreamHandler()
#     streamHandler.setFormatter(formatter)
#
#     l.setLevel(level)
#     l.addHandler(fileHandler)
#     l.addHandler(streamHandler)
#
#
# def setup_stream_logger(logger_name, level=logging.INFO):
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(level)
#
#     formatter = logging.Formatter('%(message)s')
#     streamHandler = logging.StreamHandler()
#     streamHandler.setFormatter(formatter)
#
#     logger.addHandler(streamHandler)

import unittest
class TestTempUtils(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTempUtils, self).__init__(*args, **kwargs)
        self.pretrained_embedding_file = '/Users/fei-c/Resources/embed/giga-aacw.d200.bin'

    def test_load_pre(self):
        pre_vectors, word2ix = load_pre(self.pretrained_embedding_file, binary=True, addZeroPad=True)