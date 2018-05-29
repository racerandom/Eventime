from TempMention import Token, Mention, Event, Timex, Signal
from TempLink import TimeMLDoc, TempLink
from TempUtils import *

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



def load_doc(pickle_file):
    with open(pickle_file, 'rb') as f:
        doc_list = pickle.load(f)
    return doc_list


def prepare_data(doc_dic, data_set, word_idx, pos_idx, rel_idx, max_len, types=['Event-Timex', 'Timex-Event']):
    words, pos, rels = [], [], []
    for doc_id, doc in doc_dic.items():
        if doc_id not in data_set:
            continue
        for tlink in doc.tlinks:
            if tlink.category in types:
                words.append(tlink.interwords)
                pos.append(tlink.interpos)
                rels.append(tlink.rel)
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


def prepare_global(is_pretrained=True, types=['Event-Timex', 'Timex-Event']):
    # embed_file = os.path.join(os.getenv("HOME"), 'Resources/embed/deps.words.bin')
    embed_file = os.path.join(os.getenv("HOME"), 'Resources/embed/giga-aacw.d200.bin')
    doc_dic = load_doc('data/doc_list.pkl')
    max_len = max_length(doc_dic, types)
    pos_idx = pos2idx(doc_dic)
    if is_pretrained:
        pre_model, word_idx = load_pre(embed_file)
    else:
        pre_model = None
        word_idx = word2idx(doc_dic)
    rel_idx = rel2idx(doc_dic)
    print('max seq length:', max_len, ', vocab size:', len(word_idx), ', position size:', len(pos_idx),
          ', relation categories:', len(rel_idx))
    return doc_dic, word_idx, pos_idx, rel_idx, max_len, pre_model



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