from TempObject import *
from TempUtils import *
from TimeMLReader import *

from collections import defaultdict
import random
import os
import pickle
import torch
import torch.utils.data as Data


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


TA_TEST = [ 'APW19980227.0489',
            'APW19980227.0494',
            'APW19980308.0201',
            'APW19980418.0210',
            'CNN19980126.1600.1104',
            'CNN19980213.2130.0155',
            'NYT19980402.0453',
            'PRI19980115.2000.0186',
            'PRI19980306.2000.1675']


task_feats = {'basic': ['full_word_sent',
                        'full_char_sent',
                        'sour_dist_sent',
                        'targ_dist_sent'],
              'Event-DCT': ['sour_word_seq',
                            'sour_char_seq',
                            'sour_pos_seq',
                            'sour_dep_seq',
                            'sour_index_seq',
                            'sour_word_tok',
                            'sour_char_tok',
                            'sour_pos_tok',
                            'sour_dep_tok',
                            ],
              'Event-Timex': [],
              'Event-Event': []}

def common_keys(dict1, dict2, lowercase):
    common_dict = {}
    for key in dict1.keys():
        if lowercase:
            key = key.lower()
        if key in dict2:
            common_dict[key] = len(common_dict)
    return common_dict


def batch_to_device(inputs, device):
    """
    copy input list into device for GPU computation
    :param inputs:
    :param device:
    :return:
    """
    device_inputs = []
    for input in inputs:
        device_inputs.append(input.to(device=device))
    return device_inputs


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
        data = pickle.load(f)
    print("Successfully load data from pickle file '%s'..." % pickle_file)
    return data


def save_doc(data, pickle_file='data/doc_list.pkl'):

    with open(pickle_file, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(data, f)
    print("Successfully save '%s'..." % pickle_file)


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


def readPretrainedEmbedding(pretrained_file):
    if pretrained_file and os.path.isfile(os.path.join(os.getenv("HOME"), pretrained_file)):
        pre_model, word_idx = load_pre(os.path.join(os.getenv("HOME"), pretrained_file))
    return pre_model, word_idx


def slimEmbedding(embedding_file, pickle_file, word_idx, lowercase=False):
    pre_lookup_table, pre_word_idx = readPretrainedEmbedding(embedding_file)
    common_idx = common_keys(word_idx, pre_word_idx, lowercase)
    if lowercase:
        idx = [pre_word_idx[k.lower()] for k, v in sorted(common_idx.items(), key=lambda x: x[1])
               if k.lower() in pre_word_idx]
        uncover = len(word_idx) - len(idx)
    else:
        idx = [pre_word_idx[k] for k, v in sorted(common_idx.items(), key=lambda x: x[1])
               if k in pre_word_idx]
        uncover = len(word_idx) - len(idx)
    lookup_table = pre_lookup_table[idx]
    print("[Embedding] slim embedding from %i to %i, %i tokens are uncovered..." % (pre_lookup_table.shape[0],
                                                                                    lookup_table.shape[0],
                                                                                    uncover))
    save_doc((common_idx, lookup_table), pickle_file)
    assert len(common_idx) == lookup_table.shape[0]
    return common_idx, lookup_table


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

        ## retrieve feats from link objects
        feat = []
        for doc_id, doc in doc_dic.items():
            if doc_id not in dataset:
                continue
            for link in doc.get_links_by_type(link_type):
                feat.append(link.feat_inputs[feat_type])

        ## initialize feats as tensors
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


def feat2tensorSDP(doc_dic, dataset, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
                   max_sent_len, max_seq_len, max_mention_len, max_word_len, task):

    tensor_dict = {}
    target_list = []

    feat_dict = defaultdict(list)

    for doc_id, doc in doc_dic.items():
        if doc_id not in dataset:
            continue
        if task == 'day_len':
            for event in doc.events.values():
                for feat_name, feats in event.feat_inputs.items():
                    feat_dict[feat_name].append(feats)
                target_list.append(event.daylen)
        elif task in ['Event-DCT', 'Event-Timex', 'Event-Event']:
            for link in doc.get_links_by_type(task):
                for feat_name, feats in link.feat_inputs.items():
                    feat_dict[feat_name].append(feats)
                target_list.append(link.rel)

    # initialize feats as tensors
    for feat_name, feats in feat_dict.items():
        if feat_name in ['sour_word_seq', 'targ_word_seq']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(prepare_seq_2d(feats, word_idx), max_seq_len))
        elif feat_name in ['sour_char_seq', 'targ_char_seq']:
            tensor_dict[feat_name] = torch.tensor(padding_3d(prepare_seq_3d(feats, char_idx), max_seq_len, max_word_len))
        elif feat_name in ['sour_pos_seq', 'targ_pos_seq']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(prepare_seq_2d(feats, pos_idx), max_seq_len))
        elif feat_name in ['sour_dep_seq', 'targ_dep_seq']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(prepare_seq_2d(feats, dep_idx), max_seq_len))
        elif feat_name in ['sour_word_tok', 'targ_word_tok']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(prepare_seq_2d(feats, word_idx), max_mention_len))
        elif feat_name in ['sour_char_tok', 'targ_char_tok']:
            tensor_dict[feat_name] = torch.tensor(padding_3d(prepare_seq_3d(feats, char_idx), max_mention_len, max_word_len))
        elif feat_name in ['sour_pos_tok', 'targ_pos_tok']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(prepare_seq_2d(feats, pos_idx), max_mention_len))
        elif feat_name in ['sour_dep_tok', 'targ_dep_tok']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(prepare_seq_2d(feats, dep_idx), max_mention_len))
        elif feat_name in ['full_word_sent']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(prepare_seq_2d(feats, word_idx), max_sent_len))
        elif feat_name in ['full_char_sent']:
            tensor_dict[feat_name] = torch.tensor(padding_3d(prepare_seq_3d(feats, char_idx), max_sent_len, max_word_len))
        elif feat_name in ['sour_dist_sent', 'targ_dist_sent']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(prepare_seq_2d(feats, dist_idx), max_sent_len))
        elif feat_name in ['sour_index_seq', 'targ_index_seq']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(feats, max_seq_len))
            # print(tensor_dict[feat_name])
        else:
            print("ERROR feat name: %s" % feat_name)

    target_tensor = torch.tensor(prepare_seq_1d(target_list, rel_idx))
    return tensor_dict, target_tensor


def feats2tensor_dict(doc_dic, dataset, word_idx, char_idx, dist_idx, rel_idx, max_seq_len, max_tok_len, max_char_len, link_type):

    tensor_dict = {}

    feat_types = doc_dic.values()[0].get_links_by_type(link_type)[0].keys()

    for feat_type in feat_types:

        ## retrieve feats from link objects
        feat = doc2featList(doc_dic, feat_type, link_type)

        ## initialize feats as tensors
        if feat_type in ['word_seq']:
            tensor_dict[feat_type] = torch.tensor(padding_2d(prepare_seq_2d(feat, word_idx), max_seq_len))
        elif feat_type in ['char_seq']:
            tensor_dict[feat_type] = torch.tensor(padding_3d(prepare_seq_3d(feat, char_idx), max_seq_len, max_char_len))
        elif feat_type in ['sour_dist_seq', 'targ_dist_seq']:
            tensor_dict[feat_type] = torch.tensor(padding_2d(prepare_seq_2d(feat, dist_idx), max_seq_len))
        elif feat_type in ['sour_word_tok', 'targ_word_tok']:
            tensor_dict[feat_type] = torch.tensor(padding_2d(prepare_seq_2d(feat, word_idx), max_tok_len))
        elif feat_type in ['sour_dist_tok', 'targ_dist_tok']:
            tensor_dict[feat_type] = torch.tensor(padding_2d(prepare_seq_2d(feat, dist_idx), max_tok_len))
        else:
            print("ERROR feat type: %s" % feat_type)

    target_list = []
    for doc_id, doc in doc_dic.items():
        if doc_id not in dataset:
            continue
        for link in doc.get_links_by_type(link_type):
            target_list.append(link.rel)
    target_tensor = torch.tensor(prepare_seq_1d(target_list, rel_idx))
    return tensor_dict, target_tensor


def prepareGlobalSDP(pkl_file, task):
    """
        Store features into the feat_inputs of each tlink in doc_dic.
        return doc_dic, feat to index, max length for the following process of preparing tensor inputs.
    :param pkl_file:    the pickled file of doc_dic
    :param task: default value ['Event-DCT', 'Event-Timex', 'Event-Event', 'day_len']
    :return: a tuple of doc_dic, feat to index, max length for padding
    """

    doc_dic = load_doc(pkl_file)

    """
    step1: add feats into link.feat_inputs
    """
    for doc_id, doc in doc_dic.items():
        # if doc_id != "APW19980227.0489":
        #     continue
        print("Preparing Global SDP feats for doc", doc_id)
        if task in ['day_len']:
            for event in doc.events.values():
                sent_tokens = doc.geneSentOfMention(event)
                event.feat_inputs['full_word_sent'] = [tok.content for tok in sent_tokens]
                # event.feat_inputs['full_char_sent'] = [list(tok.content.lower()) for tok in sent_tokens]
                event.feat_inputs['sour_dist_sent'] = getMentionDist(sent_tokens, event)

        elif task in ['Event-DCT', 'Event-Timex', 'Event-Event']:
            for link in doc.temp_links[task]:

                """
                add interval tokens between mentions for 'Event-Timex', 'Event-Event' tlinks
                """
                if link.link_type in ['Event-Timex', 'Event-Event']:
                    sent_tokens = doc.geneSentTokens(link.sour, link.targ)
                elif link.link_type in ['Event-DCT']:
                    sent_tokens = doc.geneSentOfMention(link.sour)
                link.feat_inputs['full_word_sent'] = [tok.content for tok in sent_tokens]
                link.feat_inputs['full_char_sent'] = [list(tok.content.lower()) for tok in sent_tokens]

                """
                add sour branch sdp 
                """
                sour_sdp_ids, sour_mention_ids, dep_graph = doc.getSdpFromMentionToRoot(link.sour)
                sour_word_seq, sour_pos_seq, sour_dep_seq = doc.getSdpFeats(sour_sdp_ids, dep_graph)
                sour_word_tok, sour_pos_tok, sour_dep_tok = doc.getSdpFeats(sour_mention_ids, dep_graph)
                link.feat_inputs['sour_dist_sent'] = getMentionDist(sent_tokens, link.sour)
                link.feat_inputs['sour_word_seq'] = sour_word_seq
                link.feat_inputs['sour_char_seq'] = [list(tok.lower()) for tok in sour_word_seq]
                link.feat_inputs['sour_pos_seq'] = sour_pos_seq
                link.feat_inputs['sour_dep_seq'] = sour_dep_seq
                link.feat_inputs['sour_word_tok'] = sour_word_tok
                link.feat_inputs['sour_char_tok'] = [list(tok.lower()) for tok in sour_word_tok]
                link.feat_inputs['sour_pos_tok'] = sour_pos_tok
                link.feat_inputs['sour_dep_tok'] = sour_dep_tok
                link.feat_inputs['sour_index_seq'] = sour_sdp_ids

                """
                add targ branch sdp for 'Event-Timex', 'Event-Event' tlinks
                """
                if link.link_type in ['Event-Timex', 'Event-Event']:
                    targ_sdp_ids, targ_mention_ids, dep_graph = doc.getSdpFromMentionToRoot(link.targ)
                    targ_word_seq, targ_pos_seq, targ_dep_seq = doc.getSdpFeats(targ_sdp_ids, dep_graph)
                    targ_word_tok, targ_pos_tok, targ_dep_tok = doc.getSdpFeats(targ_mention_ids, dep_graph)
                    link.feat_inputs['targ_dist_sent'] = getMentionDist(sent_tokens, link.targ)
                    link.feat_inputs['targ_word_seq'] = targ_word_seq
                    link.feat_inputs['targ_char_seq'] = [list(tok.lower()) for tok in targ_word_seq]
                    link.feat_inputs['targ_pos_seq'] = targ_pos_seq
                    link.feat_inputs['targ_dep_seq'] = targ_dep_seq
                    link.feat_inputs['targ_word_tok'] = targ_word_tok
                    link.feat_inputs['targ_char_tok'] = [list(tok.lower()) for tok in targ_word_tok]
                    link.feat_inputs['targ_pos_tok'] = targ_pos_tok
                    link.feat_inputs['targ_dep_tok'] = targ_dep_tok
                    link.feat_inputs['targ_index_seq'] = reviceSdpWithSentID(sent_tokens, targ_sdp_ids)


    """
    step2: calculate vocab (set object)
    """
    # word_vocab = doc2fvocab(doc_dic, 'sour_word_seq', link_types)

    pos_vocab = doc2fvocab2(doc_dic, task, ['sour_pos_seq', 'targ_pos_seq'])
    dep_vocab = doc2fvocab2(doc_dic, task, ['sour_dep_seq', 'targ_dep_seq'])
    dist_vocab = doc2fvocab2(doc_dic, task, ['sour_dist_sent', 'targ_dist_sent'])


    """
    full sentence
    """
    # word_vocab.union(doc2fvocab(doc_dic, 'full_word_seq', link_types))
    word_vocab = doc2wvocab(doc_dic)

    char_vocab = wvocab2cvocab(word_vocab)

    """
    step3: create a dict from vocab to idx
    create word index map or pre-trained embedding
    """
    word_idx = vocab2idx(word_vocab, feat_idx={'zeropadding': 0})
    char_idx = vocab2idx(char_vocab, feat_idx={'zeropadding': 0})
    pos_idx = vocab2idx(pos_vocab, feat_idx={'zeropadding': 0})
    dep_idx = vocab2idx(dep_vocab, feat_idx={'zeropadding': 0})
    dist_idx = vocab2idx(dist_vocab, feat_idx={'zeropadding': 0})

    rel_idx = rel2idx(doc_dic, task)  # target index

    """
    step4: calculate max length for padding
    """
    max_sent_len = max_length(doc_dic, task, ['full_word_sent'])
    max_seq_len = max_length(doc_dic, task, ['sour_word_seq', 'targ_word_seq'])
    max_word_len = max([len(word) for word in word_vocab])
    max_mention_len = max_length(doc_dic, task, ['sour_word_tok', 'targ_word_tok'])

    print('word vocab size: %i, '
          'char vocab size: %i, '
          'pos size: %i, '
          'dep size: %i, '
          'dist size: %i, '
          'relation size: %i' % (len(word_idx) if word_idx else 0,
                                 len(char_idx) if char_idx else 0,
                                 len(pos_idx) if pos_idx else 0,
                                 len(dep_idx) if dep_idx else 0,
                                 len(dist_idx) if dist_idx else 0,
                                 len(rel_idx) if rel_idx else 0))

    print('max sent len: %i, '
          'max word len of seq: %i , '
          'max char len of word: %i, '
          'max word len of mention: %i' % (max_sent_len,
                                           max_seq_len,
                                           max_word_len,
                                           max_mention_len))

    return doc_dic, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx, \
           max_sent_len, max_seq_len, max_mention_len, max_word_len


def prepare_DRL_data(doc_dic, dataset, word_idx, char_idx, dist_idx, rel_idx, max_seq_len, max_tok_len, max_char_len, feat_types=None):

    event_link_data = []

    for doc_id, doc in doc_dic.items():

        if doc_id not in dataset:
            continue

        for event_id, event in doc.events.items():
            event_links = []

            for link in doc.getLinkByMention(event_id):
                link_feat = []

                for feat_type in feat_types:
                    if feat_type in ['word_seq']:
                        link_feat.append(torch.tensor(padding_2d(prepare_seq_2d(feat, word_idx), max_seq_len)))
                    elif feat_type in ['char_seq']:
                        link_feat.append(
                            torch.tensor(padding_3d(prepare_seq_3d(feat, char_idx), max_seq_len, max_char_len)))
                    elif feat_type in ['sour_dist_seq', 'targ_dist_seq']:
                        link_feat.append(torch.tensor(padding_2d(prepare_seq_2d(feat, dist_idx), max_seq_len)))
                    elif feat_type in ['sour_word_tok', 'targ_word_tok']:
                        link_feat.append(torch.tensor(padding_2d(prepare_seq_2d(feat, word_idx), max_tok_len)))
                    elif feat_type in ['sour_dist_tok', 'targ_dist_tok']:
                        link_feat.append(torch.tensor(padding_2d(prepare_seq_2d(feat, dist_idx), max_tok_len)))
                    else:
                        print("ERROR feat type: %s" % feat_type)


def loadTimeML2str(timeml_file):
    str_list = []
    doc = load_anchorml(timeml_file)
    doc.setSentIds2mention()
    for index, token in enumerate(doc.tokens):
        is_event, is_timex = False, False
        for event in doc.events.values():
            if token.tok_id in event.tok_ids:
                if token.tok_id == event.tok_ids[0]:
                    str_list.append("<%s>" % event.eid)
                str_list.append(token.content)
                if token.tok_id == event.tok_ids[-1]:
                    str_list.append("</%s>" % event.eid)
                is_event = True
                continue
        for timex in doc.timexs.values():
            if token.tok_id in timex.tok_ids:
                if token.tok_id == timex.tok_ids[0]:
                    str_list.append("<T v='%s'>" % timex.value)
                str_list.append(token.content)
                if token.tok_id == timex.tok_ids[-1]:
                    str_list.append('</T>')
                is_timex = True
                continue
        if not is_event and not is_timex:
            if index > 0 and token.sent_id != doc.tokens[index - 1].sent_id:
                str_list.append('\n')
            str_list.append(token.content)
    str_out = ""
    for index, str_e in enumerate(str_list):
        if index > 0 and (str_list[index - 1].startswith('<e') or str_list[index - 1].startswith('<T')):
            str_out += str_e
        else:
            if str_e.startswith('</e') or str_e.startswith('</T'):
                str_out += str_e
            else:
                str_out += " " + str_e
    return str_out


def writeTimeML2txt(timeml_file, out_dir):
    str_out = loadTimeML2str(timeml_file)
    with open("%s/%s.txt" % (out_dir, timeml_file.split('/')[-1]), 'w') as fo:
        fo.write(str_out)


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
            doc.setSentIds2mention()  # set a sent_id to each mention and token in a doc
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


## 1. load event anchor from an aditional file and return a doc object
## 2. normalize the tanchors of all the timex entities
## 3. normalize the tanchors of all the events
## 4. induce relations of mention pairs
def anchor_file2doc(timeml_dir, anchor_file, pkl_out, sent_win, oper=False):
    anchor_docs = defaultdict(dict)
    doc_dic = {}
    with open(anchor_file, 'r') as anchor_fi:
        for line in anchor_fi:
            toks = line.strip().split()
            anchor_docs[toks[0]][toks[4]] = toks[7]
    for doc_id in anchor_docs.keys():
        filename = os.path.join(timeml_dir, doc_id + '.tml')
        try:
            doc = load_anchorml(filename)

            ## read event anchor from 'anchor_docs'
            for event_id, event in doc.events.items():
                event.value = anchor_docs[doc_id][event_id]
            doc.setSentIds2mention()  # set a sent_id to each mention and token in a doc

            ## normalize time anchors for entities
            doc.normalize_timex_value()
            doc.normalize_event_value()

            ## Induce temporal relation for E-D, E-T, E-E pairs
            doc.geneEventDCTPair(oper=oper)
            doc.geneEventTimexPair(sent_win, order='nonfixed', oper=oper)
            # doc.geneEventsPair(sent_win, oper=oper)
            doc_dic[doc.docid] = doc
        except Exception as ex:
            traceback.print_exc()
            print(filename, ex)
    save_doc(doc_dic, pkl_out)


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

    # create word index map or pre-trained embedding
    if pretrained_file and os.path.isfile(os.path.join(os.getenv("HOME"), pretrained_file)):
        pre_model, word_idx = load_pre(os.path.join(os.getenv("HOME"), pretrained_file))
    else:
        pre_model = None
        word_idx = vocab2idx(word_vocab, feat_idx={'zeropadding': 0})
        # word_idx = feat2idx(doc_dic, 'token_seq', link_type, feat_idx={'zeropadding': 0})

    # create feat index map
    if link_type in ['Event-Timex', 'Event-Event']:
        sour_dist_vocab = doc2fvocab(doc_dic, 'sour_dist_seq', link_type)
        sour_dist_idx = vocab2idx(sour_dist_vocab, feat_idx={'zeropadding': 0})
        targ_dist_vocab = doc2fvocab(doc_dic, 'targ_dist_seq', link_type)
        dist_idx = vocab2idx(targ_dist_vocab, feat_idx=sour_dist_idx)
        max_tok_len = max(max_length(doc_dic, 'sour_word_tok', link_type), max_length(doc_dic, 'targ_word_tok', link_type))
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


def writeCommonTimeML2txt(annotator1_dir, annotator2_dir, out_dir):
    annotator1_dir = os.path.join(os.path.dirname(__file__), annotator1_dir)
    annotator2_dir = os.path.join(os.path.dirname(__file__), annotator2_dir)

    common_files = list(set(os.listdir(annotator1_dir)) & set(os.listdir(annotator2_dir)))

    for filename in common_files:
        annotator1_file = os.path.join(annotator1_dir, filename)
        writeTimeML2txt(annotator1_file, out_dir)


def sample_train(full, dev, test, rate=1.0, seed=123):
    train = list(set(full) - set(dev) - set(test))
    random.seed(seed)
    random.shuffle(train)
    return train[: int(len(train) * rate)]


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

    def test_pickle_data(self):
        anchorml = "/Users/fei-c/Resources/timex/Release0531/ALL"
        link_type = 'Event-Timex'
        pkl_file = "data/unittest_%s.pkl" % (link_type)
        pickle_doc(anchorml, pkl_file, link_type)

    def test_anchor_file2doc(self):
        sent_win = 0
        oper = True
        timeml_dir = os.path.join(os.path.dirname(__file__), "data/Timebank")
        anchor_file = os.path.join(os.path.dirname(__file__), "data/event-times_normalized.tab")
        pkl_file = os.path.join(os.path.dirname(__file__),
                                "data/unittest-%s-%s_w%i_%s.pkl" % (timeml_dir.split('/')[-1],
                                                                    anchor_file.split('/')[-1],
                                                                    sent_win,
                                                                    'oper' if oper else 'order'))
        # anchor_file2doc(timeml_dir, anchor_file, pkl_file, sent_win, oper=True)

        doc_dic = load_doc(pkl_file)

        # doc = doc_dic['ABC19980304.1830.1636']

        e_d = defaultdict(lambda: 0)
        e_t = defaultdict(lambda: 0)

        for doc in doc_dic.values():
            for event_id, event in doc.events.items():
                # print(event_id, event.value)
                for link in doc.getTlinkListByMention(event_id):
                    if not link.sour.tanchor:
                        continue
                    if link.link_type == 'Event-DCT':
                        e_d[link.rel] += 1
                    if link.link_type == 'Event-Timex':
                        e_t[link.rel] += 1

        print(len(e_d.keys()), len(e_t.keys()))
        sum_e_d = sum(e_d.values())
        sum_e_t = sum(e_t.values())
        print(sum_e_d, sum_e_t)
        print()

        print("Event-DCT:")
        for key, value in sorted(e_d.items(), key=lambda kv: kv[1], reverse=True):
            print(key, "%.2f" % (value/sum_e_d))
        print()

        print("Event-Timex:")
        for key, value in sorted(e_t.items(), key=lambda kv: kv[1], reverse=True):
            print(key, "%.2f" % (value/sum_e_t))

    def test_loadTimeML2str(self):
        timeml_file = "/Users/fei-c/Resources/timex/20180919/kd/AQA024_APW199980817.1193.tml"
        print(loadTimeML2str(timeml_file))
        writeTimeML2txt(timeml_file)

    def test_writeCommonTimeML2txt(self):
        writeCommonTimeML2txt("/Users/fei-c/Resources/timex/20180919/kd", "/Users/fei-c/Resources/timex/20180919/td", "data/ref")

    def test_prepareGlobalSDP(self):

        sent_win = 1
        link_type = 'Event-DCT'
        timeml_dir = os.path.join(os.path.dirname(__file__), "data/Timebank")
        anchor_file = os.path.join(os.path.dirname(__file__), "data/event-times_normalized.tab")
        pkl_file = os.path.join(os.path.dirname(__file__), "data/unittest-%s-%s_w%i.pkl" % (timeml_dir.split('/')[-1],
                                                                                            anchor_file.split('/')[-1],
                                                                                            sent_win))

        # anchor_file2doc(timeml_dir, anchor_file, pkl_file, sent_win, oper=False)

        doc_dic, word_idx, char_idx, pos_idx, dep_idx, rel_idx, \
        max_seq_len, max_mention_len, max_word_len = prepareGlobalSDP(pkl_file, link_types=['Event-DCT', 'Event-Timex'])

        pretrained_file = "Resources/embed/giga-aacw.d200.bin"
        pre_lookup_table, word_idx = readPretrainedEmbedding(pretrained_file)
        print(pre_lookup_table.shape)
        '''
        print train/dev/test feats shape
        '''
        train_feats, train_target = feat2tensorSDP(doc_dic, TBD_TRAIN, word_idx, char_idx, pos_idx, dep_idx, rel_idx,
                                                   max_seq_len, max_mention_len, max_word_len, link_type)
        print(len(train_feats))
        print(train_feats['sour_word_seq'].shape)
        print(train_feats['sour_char_seq'].shape)
        print(train_feats['sour_word_tok'].shape)

        dev_feats, dev_target = feat2tensorSDP(doc_dic, TBD_DEV, word_idx, char_idx, pos_idx, dep_idx, rel_idx,
                                               max_seq_len, max_mention_len, max_word_len, link_type)
        print(len(dev_feats))
        print(dev_feats['sour_word_seq'].shape)
        print(dev_feats['sour_char_seq'].shape)
        print(train_feats['sour_word_tok'].shape)

        test_feats, test_target = feat2tensorSDP(doc_dic, TBD_TEST, word_idx, char_idx, pos_idx, dep_idx, rel_idx,
                                                 max_seq_len, max_mention_len, max_word_len, link_type)
        print(len(test_feats))
        print(test_feats['sour_word_seq'].shape)
        print(test_feats['sour_char_seq'].shape)
        print(train_feats['sour_word_tok'].shape)