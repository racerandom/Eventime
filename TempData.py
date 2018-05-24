from TempMention import Token, Mention, Event, Timex, Signal
from TempLink import TimeMLDoc, TempLink
import TempUtils

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


def prepare_sequence(seq_2d, to_ix):
    idxs = [ [ to_ix[w] for w in seq_1d ] for seq_1d in seq_2d]
    return idxs


def prepare_sequence_pos(seq_2d, to_ix):
    idxs = [ [ [to_ix[p_l], to_ix[p_r]] for p_l, p_r in seq_1d ] for seq_1d in seq_2d]
    return idxs

def prepare_sequence_rel(seq_1d, to_ix):
    idxs = [ to_ix[rel] for rel in seq_1d ]
    return idxs

def padding(seq_2d, max_len, pad=0, direct='right'):

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


def pos2idx(doc_dic):
    tok_idx = {'UNK':0}
    for doc in doc_dic.values():
        for tlink in doc.tlinks:
            for tok_l, tok_r in tlink.interpos:
                tok_idx.setdefault(tok_l, len(tok_idx))
                tok_idx.setdefault(tok_r, len(tok_idx))
    return tok_idx


def word2idx(doc_dic):
    tok_idx = {'UNK':0}
    for doc in doc_dic.values():
        for tlink in doc.tlinks:
            for tok in tlink.interwords:
                tok_idx.setdefault(tok, len(tok_idx))
    return tok_idx


def rel2idx(doc_dic):
    tok_idx = {}
    for doc in doc_dic.values():
        for tlink in doc.tlinks:
            tok_idx.setdefault(tlink.rel, len(tok_idx))
    return tok_idx


def max_length(doc_dic):
    word_list = []
    for doc in doc_dic.values():
        for tlink in doc.tlinks:
            word_list.append(len(tlink.interwords))
    return max(word_list)

def load_doc(pickle_file):
    with open(pickle_file, 'rb') as f:
        doc_list = pickle.load(f)
    return doc_list


def prepare_data(doc_dic, file_list, types=['Event-Timex', 'Timex-Event']):
    words, pos, rels = [], [], []
    for doc_id, doc in doc_dic.items():
        if doc_id not in file_list:
            continue
        for tlink in doc.tlinks:
            if tlink.category in types:
                words.append(tlink.interwords)
                pos.append(tlink.interpos)
                rels.append(tlink.rel)
    return words, pos, rels

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

def prepare():
    doc_dic = load_doc('data/doc_list.pkl')
    max_len = max_length(doc_dic)
    pos_idx = pos2idx(doc_dic)
    word_idx = word2idx(doc_dic)
    rel_idx = rel2idx(doc_dic)

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
    return train_w_in, train_p_in, train_r_in, max_len, pos_idx, word_idx, rel_idx

def main():
    doc_dic = load_doc('data/doc_list.pkl')
    max_len = max_length(doc_dic)
    pos_idx = pos2idx(doc_dic)
    word_idx = word2idx(doc_dic)
    rel_idx = rel2idx(doc_dic)

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
    print(train_w_in.size(), train_p_in.size(), train_r_in.unsqueeze(1).size())
    print(train_w_in.dtype, train_p_in.dtype, train_r_in.unsqueeze(1).dtype)




if __name__ == '__main__':
    main()