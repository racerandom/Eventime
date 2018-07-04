from TempMention import Token, Mention, Event, Timex, Signal
from TempLink import TimeMLDoc, TempLink
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

def prepare_data(doc_dic, data_set, word_idx, pos_idx, rel_idx, max_len, link_type):
    words, pos, rels = [], [], []

    for doc_id, doc in doc_dic.items():
        if doc_id not in data_set:
            continue
        # print(doc_id)
        if link_type in ['Event-Timex', 'Timex-Event']:
            links = doc.event_timex
        elif link_type in ['Event-DCT']:
            links = doc.event_dct
        elif link_type in ['Event-Event']:
            links = doc.event_event

        for link in links:
            words.append(link.interwords)
            pos.append(link.interpos)
            rels.append(link.rel)
            # print(link.sour.content, link.targ.content)
            # print(link.interwords)
            # print(link.interpos)
            # print(link.rel)
            # print('*' * 80)
        # print([ tok.content for tok in doc.tokens if "'" in tok.content ])
    train_w_in = torch.tensor(padding(prepare_sequence(words, word_idx), max_len))
    train_p_in = torch.tensor(padding_pos(prepare_sequence_pos(pos, pos_idx), max_len))
    train_r_in = torch.tensor(prepare_sequence_rel(rels, rel_idx))
    return train_w_in, train_p_in, train_r_in

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
def doc2pkl(anchorml_dir, pkl_file, link_type, sent_win=1):
    anchorml_list = [os.path.join(anchorml_dir, filename) for filename in sorted(os.listdir(anchorml_dir))]
    doc_dic, label_dic = {}, defaultdict(dict)
    non_count = 0
    for filename in anchorml_list:
        try:
            doc = load_anchorml(filename)
        except Exception as ex:
            traceback.print_exc()
            print(filename, ex)
        doc.setSentIds2mention()  # set a sent_id to each mention in a doc
        doc.normalize_timex_value()
        doc.normalize_event_value()
        for event in doc.events.values():
            if not event.tanchor:
                non_count += 1
        if link_type in ['ALL', 'all']:
            doc.geneEventDCTPair()
            doc.geneEventTimexPair(sent_win)
            doc.geneEventsPair(sent_win)
            for link in doc.event_dct:
                label_dic['Event-DCT'][link.rel] = label_dic['Event-DCT'].setdefault(link.rel, 0) + 1
            for link in doc.event_timex:
                label_dic['Event-Timex'][link.rel] = label_dic['Event-Timex'].setdefault(link.rel, 0) + 1
            for link in doc.events:
                label_dic['Event-Event'][link.rel] = label_dic['Event-Event'].setdefault(link.rel, 0) + 1
        elif link_type in ['Event-DCT']:
            doc.geneEventDCTPair()
            for link in doc.event_dct:
                label_dic[link_type][link.rel] = label_dic[link_type].setdefault(link.rel, 0) + 1
        elif link_type in ['Event-Timex', 'Timex-Event']:
            doc.geneEventTimexPair(sent_win)
            for link in doc.event_timex:
                label_dic[link_type][link.rel] = label_dic[link_type].setdefault(link.rel, 0) + 1
        elif link_type in ['Event-Event']:
            doc.geneEventsPair(sent_win)
            for link in doc.events:
                label_dic[link_type][link.rel] = label_dic[link_type].setdefault(link.rel, 0) + 1
        doc_dic[doc.docid] = doc

    save_doc(doc_dic, pkl_file)

    all_count = sum([value for value in label_dic[link_type].values()])
    print("number of links:", all_count, ", non event", non_count)
    print(label_dic)
    for label in sorted(label_dic[link_type].keys()):
        count = label_dic[link_type][label]
        print("label %s, num %i, rate %.2f%%" % (label, count, count * 100 / all_count))


def prepare(is_pretrained=False):

    doc_dic, max_len, pos_idx, word_idx, rel_idx, pre_model = prepare_global(is_pretrained)

    train_words, train_pos, train_rels = prepare_data(doc_dic, TBD_TRAIN,
                                                      types=['Event-Event']
                                                      )
    dev_words, dev_pos, dev_rels = prepare_data(doc_dic, TBD_DEV,
                                                types=['Event-Event']
                                                )
    test_words, test_pos, test_rels = prepare_data(doc_dic, TBD_TEST,
                                                   types=['Event-Event']
                                                   )

    train_w_in = torch.tensor(padding(prepare_sequence(train_words, word_idx), max_len))
    train_p_in = torch.tensor(padding_pos(prepare_sequence_pos(train_pos, pos_idx), max_len))
    train_r_in = torch.tensor(prepare_sequence_rel(train_rels, rel_idx))
    return train_w_in, train_p_in, train_r_in, max_len, pos_idx, word_idx, rel_idx, pre_model



## read the doc list from pkl and make a preparation of embedding processing.
def prepare_global(pkl_file, pretrained_file, link_type='Event-Timex'):
    doc_dic = load_doc(pkl_file)
    max_len = max_length(doc_dic, link_type)
    pos_idx = pos2idx(doc_dic, link_type)
    if pretrained_file and os.path.isfile(os.path.join(os.getenv("HOME"), pretrained_file)):
        pre_model, word_idx = load_pre(os.path.join(os.getenv("HOME"), pretrained_file))
    else:
        pre_model = None
        word_idx = word2idx(doc_dic, link_type)
    rel_idx = rel2idx(doc_dic, link_type)
    print('max seq length:', max_len, ', vocab size:', len(word_idx), ', position size:', len(pos_idx),
          ', relation categories:', len(rel_idx))
    return doc_dic, word_idx, pos_idx, rel_idx, max_len, pre_model


def sample_train(all, dev, test, rate=1.0, seed=123):
    train = list(set(all) - set(dev) - set(test))
    random.seed(seed)
    random.shuffle(train)
    return train[ : int(len(train) * rate) ]


def main():
    anchorml = "/Users/fei-c/Resources/timex/Release0531/ALL"
    link_type = 'Event-Timex'
    pkl_file = "data/0531_%s.pkl" % (link_type)
    doc2pkl(anchorml, pkl_file, link_type)
    doc_dic, word_idx, pos_idx, rel_idx, max_len, pre_model = prepare_global(pkl_file, None)
    # print(len(doc_dic), len(word_idx), len(rel_idx), max_len)
    # prepare_data(doc_dic, TA_TEST, word_idx, pos_idx, rel_idx, max_len, link_type)
    #
    # TA_TRAIN = sample_train(doc_dic.keys(), TA_DEV, TA_TEST, rate=0.5)
    # print(len(TA_TRAIN), len(TA_DEV), len(TA_TEST), (len(TA_TRAIN) + len(TA_DEV) + len(TA_TEST)))


if __name__ == '__main__':
    main()