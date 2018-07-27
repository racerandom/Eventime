from TempObject import *
from TempUtils import *
from TimeMLReader import *

from collections import defaultdict
import torch.utils.data as Data
import random

import os
import torch
import pickle

TBD_TRAIN = ['ABC19980120.1830.0957', 'APW19980213.1380', 'APW19980219.0476', 'ea980120.1830.0456',
             'APW19980227.0476', 'PRI19980121.2000.2591', 'CNN19980222.1130.0084', 'NYT19980206.0460',
             'ABC19980114.1830.0611', 'APW19980213.1320', 'CNN19980227.2130.0067', 'NYT19980206.0466',
             'PRI19980205.2000.1998', 'AP900816-0139', 'ABC19980108.1830.0711', 'PRI19980213.2000.0313',
             'APW19980213.1310', 'ABC19980304.1830.1636', 'AP900815-0044', 'PRI19980205.2000.1890',
             'APW19980227.0468', 'ea980120.1830.0071']

TBD_DEV = ['APW19980227.0487',
           'CNN19980223.1130.0960',
           'NYT19980212.0019',
           'PRI19980216.2000.0170',
           'ed980111.1130.0089']

TBD_TEST = ['APW19980227.0489',
            'APW19980227.0494',
            'APW19980308.0201',
            'APW19980418.0210',
            'CNN19980126.1600.1104',
            'CNN19980213.2130.0155',
            'NYT19980402.0453',
            'PRI19980115.2000.0186',
            'PRI19980306.2000.1675']


TA_DEV = [ 'wsj_0924',
           'APW19980626.0364',
           'wsj_0527',
           'APW19980227.0487',
           'CNN19980223.1130.0960',
           'NYT19980212.0019',
           'PRI19980216.2000.0170',
           'ed980111.1130.0089']

TA_TEST = ['APW19980227.0489',
            'APW19980227.0494',
            'APW19980308.0201',
            'APW19980418.0210',
            'CNN19980126.1600.1104',
            'CNN19980213.2130.0155',
            'NYT19980402.0453',
            'PRI19980115.2000.0186',
            'PRI19980306.2000.1675']


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


def load_doc(pickle_file):
    with open(pickle_file, 'rb') as f:
        doc_list = pickle.load(f)
    return doc_list


def save_doc(doc_data, pickle_file='data/doc_list.pkl'):

    with open(pickle_file, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(doc_data, f)
    print("Successfully save '%s'..." % pickle_file)


def prepare_dataset(doc_dic, dataset, word_idx, pos_idx, rel_idx, max_len, link_type):
    words, pos, rels = [], [], []

    for doc_id, doc in doc_dic.items():
        if doc_id not in dataset:
            continue
        for link in doc.get_links_by_type(link_type):
            words.append(link.interwords)
            pos.append(link.interpos)
            rels.append(link.rel)
    train_w_in = torch.tensor(padding(prepare_sequence(words, word_idx), max_len))
    train_p_in = torch.tensor(padding_pos(prepare_sequence_pos(pos, pos_idx), max_len))
    train_r_in = torch.tensor(prepare_sequence_rel(rels, rel_idx))
    return train_w_in, train_p_in, train_r_in


## generate a list of feat tensor and target tensor of a given dataset
def prepare_feats_dataset(doc_dic, dataset, word_idx, char_idx, dist_idx, rel_idx, max_seq_len, max_tok_len, max_char_len, link_type, feat_types=['word_seq',
                                                                                                                                'sour_dist_seq',
                                                                                                                                'targ_dist_seq',
                                                                                                                                'sour_word_tok',
                                                                                                                                'targ_word_tok',
                                                                                                                                'sour_dist_tok',
                                                                                                                                'targ_dist_tok']):

    feats_list = []

    for feat_type in feat_types:
        feat = []
        for doc_id, doc in doc_dic.items():
            if doc_id not in dataset:
                continue
            for link in doc.get_links_by_type(link_type):
                feat.append(link.feat_inputs[feat_type])

        if feat_type in ['word_seq']:
            feats_list.append(torch.tensor(padding_2d(prepare_seq_2d(feat, word_idx), max_seq_len)))
        elif feat_type in ['char_seq']:
            feats_list.append(torch.tensor(padding_3d(prepare_seq_3d(feat, char_idx), max_seq_len, max_char_len)))
        elif feat_type in ['sour_dist_seq', 'targ_dist_seq']:
            feats_list.append(torch.tensor(padding_2d(prepare_seq_2d(feat, dist_idx), max_seq_len)))
        elif feat_type in ['sour_word_tok', 'targ_word_tok']:
            feats_list.append(torch.tensor(padding_2d(prepare_seq_2d(feat, word_idx), max_tok_len)))
        elif feat_type in ['sour_dist_tok', 'targ_dist_tok']:
            feats_list.append(torch.tensor(padding_2d(prepare_seq_2d(feat, dist_idx), max_tok_len)))
        else:
            print("ERROR feat type: %s" % feat_type)

    target_list = []
    for doc_id, doc in doc_dic.items():
        if doc_id not in dataset:
            continue
        for link in doc.get_links_by_type(link_type):
            target_list.append(link.rel)
    target_tensor = torch.tensor(prepare_seq_1d(target_list, rel_idx))
    return feats_list, target_tensor


def prepare_artificial_classification():
    VOCAB_SIZE = 100
    POS_SIZE = 10
    MAX_LEN = 10
    ACTION_SIZE = 2
    dct_inputs = torch.randint(0, VOCAB_SIZE, (500, 1, MAX_LEN), dtype=torch.long)
    position_inputs = torch.randint(0, POS_SIZE, (500, MAX_LEN, 2), dtype=torch.long)
    time_inputs = torch.randint(0, VOCAB_SIZE, (500, 1, 2, 3), dtype=torch.long)

    targets = torch.randint(0, ACTION_SIZE, (500, 1), dtype=torch.long)

    return dct_inputs, position_inputs, time_inputs, targets


## 1. load anchorml files and return a doc object
## 2. normalize the tanchors of all the timex entities
## 3. normalize the tanchors of all the events
## 4. induce relations of mention pairs
def pickle_doc(anchorml_dir, pkl_file, link_type, sent_win=1):
    anchorml_list = [os.path.join(anchorml_dir, filename) for filename in sorted(os.listdir(anchorml_dir))]
    doc_dic, label_dic = {}, defaultdict(dict)
    non_count = 0
    for filename in anchorml_list:
        try:
            doc = load_anchorml(filename)
            doc.setSentIds2mention()  # set a sent_id to each mention in a doc
            doc.normalize_timex_value()
            doc.normalize_event_value()
            for event in doc.events.values():
                if not event.tanchor:
                    non_count += 1
            doc.geneEventDCTPair()
            doc.geneEventTimexPair(sent_win)
            doc.geneEventsPair(sent_win)
            for link in doc.get_links_by_type(link_type):
                label_dic[link_type][link.rel] = label_dic[link_type].setdefault(link.rel, 0) + 1
            doc_dic[doc.docid] = doc
        except Exception as ex:
            traceback.print_exc()
            print(filename, ex)

    save_doc(doc_dic, pkl_file)

    all_count = sum([value for value in label_dic[link_type].values()])
    print("number of links:", all_count, ", non event", non_count)
    print(label_dic)
    for label in sorted(label_dic[link_type].keys()):
        count = label_dic[link_type][label]
        print("label %s, num %i, rate %.2f%%" % (label, count, count * 100 / all_count))


def count_mention_nb(timeml_dir):
    timeml_list = [os.path.join(timeml_dir, filename) for filename in sorted(os.listdir(timeml_dir))]
    doc_dic, label_dic = {}, defaultdict(dict)
    for filename in timeml_list:
        try:
            doc = load_anchorml(filename)
            doc_dic[doc.docid] = doc
        except Exception as ex:
            traceback.print_exc()
            print(filename, ex)

    event_nb = sum([len(doc.events) for doc in doc_dic.values()])
    timex_nb = sum([len(doc.timexs) for doc in doc_dic.values()])
    signal_nb = sum([len(doc.signals) for doc in doc_dic.values()])
    print("Event numb: %i, Timex numb: %i, Signal numb: %i" % (event_nb, timex_nb, signal_nb) )

## 1) read the doc list from pkl and
## 2) create feat inputs for links
## 3) make a preparation of embedding processing.
def prepare_global(pkl_file, pretrained_file, link_type='Event-Timex'):

    doc_dic = load_doc(pkl_file)

    ## add feats into link.feat_inputs
    for doc in doc_dic.values():
        for link in doc.get_links_by_type(link_type):
            if link_type in ['Event-Timex', 'Event-Event']:
                tokens = doc.geneSentTokens(link.sour, link.targ)
                link.feat_inputs['word_seq'] = [tok.content for tok in tokens]
                link.feat_inputs['char_seq'] = [ list(tok.content.lower()) for tok in tokens]
                link.feat_inputs['sour_dist_seq'] = getMentionDist(tokens, link.sour)
                link.feat_inputs['targ_dist_seq'] = getMentionDist(tokens, link.targ)
                link.feat_inputs['sour_word_tok'] = link.sour.content.split()
                link.feat_inputs['targ_word_tok'] = link.targ.content.split()
                link.feat_inputs['sour_dist_tok'], link.feat_inputs['targ_dist_tok']  = getEndPosition(link.sour, link.targ)
            elif link_type in ['Event-DCT']:
                tokens = doc.geneSentOfMention(link.sour)
                link.feat_inputs['word_seq'] = [tok.content for tok in tokens]
                link.feat_inputs['char_seq'] = [list(tok.content.lower()) for tok in tokens]
                link.feat_inputs['sour_dist_seq'] = getMentionDist(tokens, link.sour)
                link.feat_inputs['sour_word_tok'] = link.sour.content.split()

    word_vocab = doc2fvocab(doc_dic, 'word_seq', link_type)
    char_vocab = wvocab2cvocab(word_vocab)
    char_idx = vocab2idx(char_vocab, feat_idx={'zeropadding': 0})
    max_char_len = max([len(word) for word in word_vocab])

    ## create word index map or pre-trained embedding
    if pretrained_file and os.path.isfile(os.path.join(os.getenv("HOME"), pretrained_file)):
        pre_model, word_idx = load_pre(os.path.join(os.getenv("HOME"), pretrained_file))
    else:
        pre_model = None
        word_idx = vocab2idx(word_vocab, feat_idx={'zeropadding': 0})
        # word_idx = feat2idx(doc_dic, 'token_seq', link_type, feat_idx={'zeropadding': 0})

    ## create feat index map
    if link_type in ['Event-Timex', 'Event-Event']:
        sour_dist_vocab = doc2fvocab(doc_dic, 'sour_dist_seq', link_type)
        sour_dist_idx = vocab2idx(sour_dist_vocab, feat_idx={'zeropadding': 0})
        targ_dist_vocab = doc2fvocab(doc_dic, 'targ_dist_seq', link_type)
        dist_idx = vocab2idx(targ_dist_vocab, feat_idx=sour_dist_idx)
        max_tok_len = max(max_length(doc_dic, 'sour_word_tok', link_type), max_length(doc_dic, 'sour_word_tok', link_type))
    elif link_type in ['Event-DCT']:
        dist_vocab = doc2fvocab(doc_dic, 'sour_dist_seq', link_type)
        dist_idx = vocab2idx(dist_vocab, feat_idx={'zeropadding': 0})
        max_tok_len = max_length(doc_dic, 'sour_word_tok', link_type)
    rel_idx = rel2idx(doc_dic, link_type)
    max_seq_len = max_length(doc_dic, 'word_seq', link_type)
    print('word vocab size: %i, char vocab size: %i, dist size: %i, relation size: %i\n' \
          'max word len of seq: %i , max char len of word: %i, , max word len of mention: %i' %  (len(word_idx),
                                                                                                 len(char_idx),
                                                                                                 len(dist_idx),
                                                                                                 len(rel_idx),
                                                                                                 max_seq_len,
                                                                                                 max_char_len,
                                                                                                 max_tok_len))
    return doc_dic, word_idx, char_idx, dist_idx, rel_idx, \
           max_seq_len, max_tok_len, max_char_len, pre_model


def sample_train(all, dev, test, rate=1.0, seed=123):
    train = list(set(all) - set(dev) - set(test))
    random.seed(seed)
    random.shuffle(train)
    return train[ : int(len(train) * rate) ]


def main():
    anchorml = "/Users/fei-c/Resources/timex/Release0531/ALL"
    link_type = 'Event-Timex'
    pkl_file = "data/0531_%s.pkl" % (link_type)
    # pickle_doc(anchorml, pkl_file, link_type)

    # timeml_dir = "/Users/fei-c/Downloads/AQ+TE3/AQUAINT"
    # count_mention_nb(timeml_dir)


    # doc_dic, word_idx, pos_idx, rel_idx, max_len, pre_model = prepare_global(pkl_file, None)
    # print(len(doc_dic), len(word_idx), len(rel_idx), max_len)
    # prepare_data(doc_dic, TA_TEST, word_idx, pos_idx, rel_idx, max_len, link_type)
    #
    # TA_TRAIN = sample_train(doc_dic.keys(), TA_DEV, TA_TEST, rate=0.5)
    # print(len(TA_TRAIN), len(TA_DEV), len(TA_TEST), (len(TA_TRAIN) + len(TA_DEV) + len(TA_TEST)))

if __name__ == '__main__':
    main()


import unittest

class TestTempData(unittest.TestCase):

    def test_prepare_global(self):
        link_type = 'Event-Timex'
        pkl_file = "data/0531_%s.pkl" % (link_type)
        doc_dic, word_idx, char_idx, dist_idx, rel_idx, \
        max_seq_len, max_token_len, max_char_len, pre_model = prepare_global(pkl_file, None)

    def test_prepare_feat_dataset(self):
        link_type = 'Event-Timex'
        pkl_file = "data/0531.pkl"
        feat_types = ['word_seq',
                    # 'char_seq',
                    'sour_dist_seq',
                    'targ_dist_seq',
                    'sour_word_tok',
                    'targ_word_tok',
                    'sour_dist_tok',
                    'targ_dist_tok']
        doc_dic, word_idx, char_idx, dist_idx, rel_idx, \
        max_seq_len, max_tok_len, max_char_len, pre_model = prepare_global(pkl_file, None, link_type)
        train_inputs, train_target = prepare_feats_dataset(doc_dic, TA_TEST, word_idx, char_idx, dist_idx, rel_idx, max_seq_len, max_tok_len, max_char_len, link_type, feat_types=feat_types)
        print([ (feat_type, feat.shape) for feat, feat_type in zip(train_inputs, feat_types)], train_target.shape)

