from TempUtils import *
from TimeMLReader import *

from collections import *
from typing import *
import random
import os
import torch


def common_keys(dict1, dict2, lowercase):
    common_dict = {}
    for key in dict1.keys():
        if lowercase:
            key = key.lower()
        if key in dict2:
            common_dict[key] = len(common_dict)
    return common_dict


class Vocabulary:

    def __init__(self, counter: Dict[str, Counter] = None) -> None:
        self._counter = counter
        self._token_to_index = None
        self._index_to_token = None

    def add_instance_to_namespace(self, instance: list, namespace: str = 'tokens') -> None:
        self._counter[namespace].update(instance)

    def index_token(self,
                    min_count: Dict[str, int] = None,
                    max_vocab_size: Dict[str, int] = None,
                    zero_unk: Dict[str, bool] = None):

        self._token_to_index = defaultdict(dict)
        self._index_to_token = defaultdict(dict)
        for namespace in self._counter.keys():
            counter_size = len(self._counter[namespace])
            filter_items = self._counter[namespace].most_common(counter_size if not max_vocab_size[namespace] else max_vocab_size[namespace])
            filter_items = [item for item in filter_items if item[1] >= min_count[namespace]]
            if zero_unk[namespace]:
                self._token_to_index[namespace]['unk_token'] = 0
            for token, freq in filter_items:
                self._token_to_index[namespace][token] = len(self._token_to_index[namespace])

            self._index_to_token[namespace] = {v: k for k, v in self._token_to_index[namespace].items()}

    def add_token_to_namespace(self, token: str, namespace: str = 'tokens') -> None:
        self._counter[namespace].update([token])

    def get_token_to_index(self):
        return self._token_to_index

    def get_index_to_token(self):
        return self._index_to_token

    def get_index_from_token(self, token: str, unk_index: int = 0, namespace: str = 'tokens') -> int:
        if token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][token]
        else:
            return unk_index

    def get_token_from_index(self, index: int, namespace: str = 'tokens') -> str:
        return self._index_to_token[namespace][index]

    def get_vocab_size(self, namespace: str = 'tokens') -> int:
        return len(self._token_to_index[namespace])

    def get_label_dist(self, max_label_size: int = 10):
        total_num = sum([c for c in self._counter['labels'].values()])
        label_size = len(self._counter['labels'])
        for label, count in self._counter['labels'].most_common(label_size if not max_label_size else max_label_size):
            print(label, count / total_num)


class DataInstance:

    def __init__(self, fields) -> None:
        self._fields = fields
        self._field_indices = None

    def count_vocab_items(self, counter: Dict[str, Counter] = None):
        if not counter:
            counter = defaultdict(Counter)
        for namespace, field in self._fields.items():
            for data_line in field:
                counter[namespace].update(data_line)
        return counter

    def index_field(self, vocab: Vocabulary) -> None:
        self._field_indices = defaultdict(list)
        for namespace, field in self._fields.items():
            for data_line in field:
                self._field_indices[namespace].append(
                    [vocab.get_index_from_token(token, namespace=namespace) for token in data_line]
                )

    def get_padding_lengths(self) -> Dict[str, List[int]]:
        padding_lengths = defaultdict(list)
        for namespace, field in self._fields.items():
            for data_line in field:
                padding_lengths[namespace].append(len(data_line))
        return padding_lengths

    def as_tensor_dict(self,
                       padding_lengths: Dict[str, List[int]] = None) -> Dict[str, torch.Tensor]:
        field_tensors = dict()
        for namespace, indice in self._field_indices.items():
            max_field_length = max(padding_lengths[namespace])
            field_tensors[namespace] = torch.tensor(padding_2d(indice, max_field_length))
        return field_tensors


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


def read_word2ix_from_doc(doc_pickle):

    doc_dic = load_from_pickle(doc_pickle)

    doc_tokens = [[token.content for token in doc.tokens] for doc in doc_dic.values()]

    word2ix = feat_to_ix(doc_tokens)

    return word2ix


def readPretrainedEmbedding(pretrained_file):
    if pretrained_file and os.path.isfile(os.path.join(os.getenv("HOME"), pretrained_file)):
        pre_model, word_idx = load_pre(os.path.join(os.getenv("HOME"), pretrained_file))
        return pre_model, word_idx
    else:
        raise Exception('[ERROR] Un-existing pretrain embedding file \'%s\'...' % pretrained_file)


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
    save_to_pickle((common_idx, lookup_table), pickle_file)
    assert len(common_idx) == lookup_table.shape[0]
    return common_idx, lookup_table


def feat2tensorSDP(doc_dic, dataset, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
                   max_sent_len, max_sdp_len, max_mention_len, max_word_len, task):

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
        if feat_name in ['sour_word_sdp', 'targ_word_sdp']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(prepare_seq_2d(feats, word_idx), max_sdp_len))
        elif feat_name in ['sour_char_sdp', 'targ_char_sdp']:
            tensor_dict[feat_name] = torch.tensor(padding_3d(prepare_seq_3d(feats, char_idx), max_sdp_len, max_word_len))
        elif feat_name in ['sour_pos_sdp', 'targ_pos_sdp']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(prepare_seq_2d(feats, pos_idx), max_sdp_len))
        elif feat_name in ['sour_dep_sdp', 'targ_dep_sdp']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(prepare_seq_2d(feats, dep_idx), max_sdp_len))
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
        elif feat_name in ['full_elmo_sent']:
            character_ids = batch_to_ids(feats)
            if character_ids.shape[1] < max_sent_len:
                character_ids = torch.cat((character_ids,
                                           torch.zeros(character_ids.shape[0],
                                                       max_sent_len - character_ids.shape[1],
                                                       character_ids.shape[-1]).long()
                                           ),
                                          dim=1)
            tensor_dict[feat_name] = character_ids
        elif feat_name in ['full_char_sent']:
            tensor_dict[feat_name] = torch.tensor(padding_3d(prepare_seq_3d(feats, char_idx), max_sent_len, max_word_len))
        elif feat_name in ['sour_dist_sent', 'targ_dist_sent']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(prepare_seq_2d(feats, dist_idx), max_sent_len))
        elif feat_name in ['sour_index_sdp', 'targ_index_sdp']:
            tensor_dict[feat_name] = torch.tensor(padding_2d(feats, max_sdp_len))
            # print(tensor_dict[feat_name])
        else:
            print("ERROR feat name: %s" % feat_name)

    target_tensor = torch.tensor(prepare_seq_1d(target_list, rel_idx))
    return tensor_dict, target_tensor


def feats2tensor_dict(doc_dic, dataset, word_idx, char_idx, dist_idx, rel_idx, max_sdp_len, max_tok_len, max_char_len, link_type):

    tensor_dict = {}

    feat_types = doc_dic.values()[0].get_links_by_type(link_type)[0].keys()

    for feat_type in feat_types:

        # retrieve feats from link objects
        feat = doc2featList(doc_dic, feat_type, link_type)

        # initialize feats as tensors
        if feat_type in ['word_sdp']:
            tensor_dict[feat_type] = torch.tensor(padding_2d(prepare_seq_2d(feat, word_idx), max_sdp_len))
        elif feat_type in ['char_sdp']:
            tensor_dict[feat_type] = torch.tensor(padding_3d(prepare_seq_3d(feat, char_idx), max_sdp_len, max_char_len))
        elif feat_type in ['sour_dist_sdp', 'targ_dist_sdp']:
            tensor_dict[feat_type] = torch.tensor(padding_2d(prepare_seq_2d(feat, dist_idx), max_sdp_len))
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


def prepare_feats(doc_dic: Dict, addSEP=None) -> Tuple[Dict[str, List],
                                                       Dict[str, List],
                                                       Tuple[Dict[str, List], Dict[str, List]],
                                                       Tuple[List, List]]:

    ed_fields, et_fields = defaultdict(list), defaultdict(list)
    ed_indices, et_indices = defaultdict(list), defaultdict(list)

    for doc_id, doc in doc_dic.items():

        for link_type in ['Event-DCT', 'Event-Timex']:

            if link_type in doc.temp_links:
                for link in doc.temp_links[link_type]:

                    event = link.sour if link.sour.mention_type == 'Event' else link.targ
                    timex = link.sour if link.sour.mention_type != 'Event' else link.targ
                    key = '%s ||| %s' % (doc_id, event.eid)

                    if link_type == 'Event-DCT':

                        ed_indices[key].append((len(ed_fields['labels']), TempUtils.norm_time_4to2(timex.tanchor)))
                        ed_fields['labels'].append(list(link.rel))

                        sent_tokens = doc.geneSentOfMention(link.sour, addSEP=addSEP)

                        if addSEP:
                            ed_fields['tokens'].append(sent_tokens)
                        else:

                            event_masks = update_entity_masks(sent_tokens, link.sour.tok_ids, mask_id=1)

                            ed_fields['tokens'].append([token.content.lower for token in sent_tokens])
                            ed_fields['event_masks'].append(event_masks)
                            ed_fields['event_dist'].append(getMentionDist(sent_tokens, link.sour, prefix='e'))

                    elif link_type == 'Event-Timex':

                        et_indices[key].append((len(et_fields['labels']), TempUtils.norm_time_4to2(timex.tanchor)))
                        et_fields['labels'].append(list(link.rel))

                        sent_tokens = doc.geneSentTokens(link.sour, link.targ, addSEP=addSEP)

                        if addSEP:
                            et_fields['tokens'].append(sent_tokens)
                        else:
                            et_fields['tokens'].append([token.content.lower for token in sent_tokens])
                            et_fields['event_dist'].append(getMentionDist(sent_tokens, event, prefix='e'))
                            et_fields['timex_dist'].append(getMentionDist(sent_tokens, timex, prefix='t'))
                            event_masks = update_entity_masks(
                                sent_tokens,
                                event.tok_ids,
                                mask_id=1
                            )
                            timex_masks = update_entity_masks(
                                sent_tokens,
                                timex.tok_ids,
                                mask_id=1,
                            )
                            et_fields['event_masks'].append(event_masks)
                            et_fields['timex_masks'].append(timex_masks)
                    else:
                        raise Exception("[ERROR] Unknown link_type parameter!!!")

    return ed_fields, et_fields, (ed_indices, et_indices), (ed_fields['labels'], et_fields['labels'])


def prepare_gold(doc_dic):

    targets = {}
    for doc_id, doc in doc_dic.items():
        for link in doc.temp_links['Event-DCT']:
            event = link.sour if link.sour.mention_type == 'Event' else link.targ
            key = '%s ||| %s' % (doc_id, event.eid)
            targets[key] = event.tanchor
    return targets


def get_vocab_and_lengths(
        train_instances: DataInstance,
        val_instances: DataInstance,
        test_instances: DataInstance,
        min_count: Dict[str, int],
        max_vocab_size: Dict[str, int],
        zero_unk: Dict[str, bool]
):

    vocab_counter = train_instances.count_vocab_items()
    vocab_counter = val_instances.count_vocab_items(vocab_counter)
    vocab_counter = test_instances.count_vocab_items(vocab_counter)

    vocab = Vocabulary(counter=vocab_counter)

    vocab.index_token(min_count=min_count,
                      max_vocab_size=max_vocab_size,
                      zero_unk=zero_unk)

    train_padding_lengths = train_instances.get_padding_lengths()
    val_padding_lengths = val_instances.get_padding_lengths()
    test_padding_lengths = test_instances.get_padding_lengths()

    padding_lengths = {
        namespace: train_padding_lengths[namespace] +
        val_padding_lengths[namespace] +
        test_padding_lengths[namespace] for namespace in train_padding_lengths.keys()
    }

    return vocab, padding_lengths


def prepareGlobalSDP(doc_pkl_file, task):
    """
        Store features into the feat_inputs of each tlink in doc_dic.
        return doc_dic, feat to index, max length for the following process of preparing tensor inputs.
    :param pkl_file:    the pickled file of doc_dic
    :param task: default value ['Event-DCT', 'Event-Timex', 'Event-Event', 'day_len']
    :return: a tuple of doc_dic, feat to index, max length for padding
    """

    doc_dic = load_from_pickle(doc_pkl_file)

    """
    step1: add feats into link.feat_inputs
    """
    for doc_id, doc in doc_dic.items():
        # if doc_id != "APW19980227.0489":
        #     continue
        print("Preparing Global SDP feats for doc", doc_id)
        if task in ['day_len']:
            for event in doc.events.values():
                # sent feats
                sent_tokens = doc.geneSentOfMention(event)
                event.feat_inputs['full_word_sent'] = [tok.content for tok in sent_tokens]
                # event.feat_inputs['full_char_sent'] = [list(tok.content.lower()) for tok in sent_tokens]
                event.feat_inputs['sour_dist_sent'] = getMentionDist(sent_tokens, event)
                event.feat_inputs['full_elmo_sent'] = [tok.content for tok in sent_tokens]
                # lexical feats
                sdp_conll_ids, event_conll_ids, dep_graph = doc.getSdpFromMentionToRoot(event)
                event_word_sdp, event_pos_sdp, event_dep_sdp = doc.getSdpFeats(sdp_conll_ids, dep_graph)
                event_word, event_pos, event_dep = doc.getSdpFeats(event_conll_ids, dep_graph)
                event.feat_inputs['sour_word_sdp'] = event_word_sdp
                event.feat_inputs['sour_pos_sdp'] = event_pos_sdp
                event.feat_inputs['sour_dep_sdp'] = event_dep_sdp
                event.feat_inputs['sour_word_tok'] = event_word
                event.feat_inputs['sour_pos_tok'] = event_pos
                event.feat_inputs['sour_dep_tok'] = event_dep
                event.feat_inputs['sour_index_sdp'] = sdp_conll_ids

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
                sour_word_sdp, sour_pos_sdp, sour_dep_sdp = doc.getSdpFeats(sour_sdp_ids, dep_graph)
                sour_word_tok, sour_pos_tok, sour_dep_tok = doc.getSdpFeats(sour_mention_ids, dep_graph)
                link.feat_inputs['sour_dist_sent'] = getMentionDist(sent_tokens, link.sour)
                link.feat_inputs['sour_word_sdp'] = sour_word_sdp
                link.feat_inputs['sour_char_sdp'] = [list(tok.lower()) for tok in sour_word_sdp]
                link.feat_inputs['sour_pos_sdp'] = sour_pos_sdp
                link.feat_inputs['sour_dep_sdp'] = sour_dep_sdp
                link.feat_inputs['sour_word_tok'] = sour_word_tok
                link.feat_inputs['sour_char_tok'] = [list(tok.lower()) for tok in sour_word_tok]
                link.feat_inputs['sour_pos_tok'] = sour_pos_tok
                link.feat_inputs['sour_dep_tok'] = sour_dep_tok
                link.feat_inputs['sour_index_sdp'] = sour_sdp_ids

                """
                add targ branch sdp for 'Event-Timex', 'Event-Event' tlinks
                """
                if link.link_type in ['Event-Timex', 'Event-Event']:
                    targ_sdp_ids, targ_mention_ids, dep_graph = doc.getSdpFromMentionToRoot(link.targ)
                    targ_word_sdp, targ_pos_sdp, targ_dep_sdp = doc.getSdpFeats(targ_sdp_ids, dep_graph)
                    targ_word_tok, targ_pos_tok, targ_dep_tok = doc.getSdpFeats(targ_mention_ids, dep_graph)
                    link.feat_inputs['targ_dist_sent'] = getMentionDist(sent_tokens, link.targ)
                    link.feat_inputs['targ_word_sdp'] = targ_word_sdp
                    link.feat_inputs['targ_char_sdp'] = [list(tok.lower()) for tok in targ_word_sdp]
                    link.feat_inputs['targ_pos_sdp'] = targ_pos_sdp
                    link.feat_inputs['targ_dep_sdp'] = targ_dep_sdp
                    link.feat_inputs['targ_word_tok'] = targ_word_tok
                    link.feat_inputs['targ_char_tok'] = [list(tok.lower()) for tok in targ_word_tok]
                    link.feat_inputs['targ_pos_tok'] = targ_pos_tok
                    link.feat_inputs['targ_dep_tok'] = targ_dep_tok
                    link.feat_inputs['targ_index_sdp'] = reviceSdpWithSentID(sent_tokens, targ_sdp_ids)


    """
    step2: calculate vocab (set object)
    """
    # word_vocab = doc2fvocab(doc_dic, 'sour_word_sdp', link_types)

    pos_vocab = doc2fvocab2(doc_dic, task, ['sour_pos_sdp', 'targ_pos_sdp', 'sour_pos_tok', 'targ_pos_tok'])
    dep_vocab = doc2fvocab2(doc_dic, task, ['sour_dep_sdp', 'targ_dep_sdp', 'sour_dep_tok', 'targ_dep_tok'])
    dist_vocab = doc2fvocab2(doc_dic, task, ['sour_dist_sent', 'targ_dist_sent'])


    """
    full sentence
    """
    # word_vocab.union(doc2fvocab(doc_dic, 'full_word_sdp', link_types))
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
    max_sdp_len = max_length(doc_dic, task, ['sour_word_sdp', 'targ_word_sdp'])
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
                                           max_sdp_len,
                                           max_word_len,
                                           max_mention_len))

    return doc_dic, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx, \
           max_sent_len, max_sdp_len, max_mention_len, max_word_len


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


def anchorML_to_doc_pickle(anchorml_dir, pkl_file):
    """
    1. load anchorml files and return a doc object
    2. normalize the tanchors of all the timex entities
    3. normalize the tanchors of all the events
    4. induce relations of mention pairs
    """

    anchorml_list = [os.path.join(anchorml_dir, filename) for filename in sorted(os.listdir(anchorml_dir))]
    doc_dic = {}
    for filename in anchorml_list:
        _, key = os.path.split(filename)
        try:
            doc = load_anchorml(filename)
            doc_dic[key] = doc
        except Exception as ex:
            traceback.print_exc()
            print(filename, ex)

    save_to_pickle(doc_dic, pkl_file)


def anchorML_to_doc(anchorml_dir):
    """
    1. load anchorml files and return a doc object
    2. normalize the tanchors of all the timex entities
    3. normalize the tanchors of all the events
    4. induce relations of mention pairs
    """

    anchorml_list = [os.path.join(anchorml_dir, filename) for filename in sorted(os.listdir(anchorml_dir))]
    doc_dic = {}
    # non_count = 0
    for filename in anchorml_list:
        try:
            doc = load_anchorml(filename)
            doc_dic[doc.docid] = doc
        except Exception as ex:
            traceback.print_exc()
            print(filename, ex)

    return doc_dic


def refine_tanchor(tanchor):
    if tanchor[1] is None and tanchor[3] is not None:
        return True
    elif tanchor[2] is None and tanchor[0] is not None:
        return True
    else:
        return False


def is_certain_tanchor(tanchor):
    if None in tanchor:
        return False
    elif tanchor[0] == tanchor[1] == tanchor[2] == tanchor[3]:
        return True
    elif tanchor[0] == tanchor[1] and tanchor[2] == tanchor[3]:
        return True
    else:
        return False


def preprocess_doc(doc_dic, oper=3, anchor_type='tanchor', sent_win=0, reverse_rel=None):

    event_count, refine_cases, certain_case = 0, 0, 0
    for doc_id, doc in doc_dic.items():
        doc.setSentIds2mention()  # set a sent_id to each mention and token in a doc
        doc.normalize_timex_value(verbose=0)
        doc.normalize_event_value(anchor_type, verbose=0)
        for e_id, event in doc.events.items():
            if event.tanchor:
                event_count += 1
                if refine_tanchor(event.tanchor):
                    refine_cases += 1
                if is_certain_tanchor(event.tanchor):
                    certain_case += 1
            else:
                print(e_id, event.value, event.tanchor)
        doc.geneEventDCTPair(oper=oper)
        doc.geneEventTimexPair(sent_win, order=False, oper=oper, reverse_rel=reverse_rel)

    print("Event num: %i" % sum([len(doc.events) for doc in doc_dic.values()]))
    print(", effective event num: %i, refined num %i, certain_case %i" % (
        event_count,
        refine_cases,
        certain_case
    ))
    return doc_dic


def anchor_file2doc(timeml_dir, anchor_file, pkl_out, sent_win, oper=False):
    """
    1. load event anchor from an aditional file and return a doc object
    2. normalize the tanchors of all the timex entities
    3. normalize the tanchors of all the events
    4. induce relations of mention pairs
    :param timeml_dir:
    :param anchor_file:
    :param pkl_out:
    :param sent_win:
    :param oper:
    :return: None
    """
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

            # read event anchor from 'anchor_docs'
            for event_id, event in doc.events.items():
                event.value = anchor_docs[doc_id][event_id]
            doc.setSentIds2mention()  # set a sent_id to each mention and token in a doc

            # normalize time anchors for entities
            doc.normalize_timex_value()
            doc.normalize_event_value()

            # Induce temporal relation for E-D, E-T, E-E pairs
            doc.geneEventDCTPair(oper=oper)
            doc.geneEventTimexPair(sent_win, order='nonfixed', oper=oper)

            # doc.geneEventsPair(sent_win, oper=oper)
            doc_dic[doc.docid] = doc
        except Exception as ex:
            traceback.print_exc()
            print(filename, ex)
    save_to_pickle(doc_dic, pkl_out)


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


# 1) read the doc list from pkl and
# 2) create feat inputs for links
# 3) make a preparation of embedding processing.
def prepare_global(pkl_file, pretrained_file, link_type='Event-Timex'):

    doc_dic = load_from_pickle(pkl_file)

    # add feats into link.feat_inputs
    for doc in doc_dic.values():
        for link in doc.get_links_by_type(link_type):
            if link_type in ['Event-Timex', 'Event-Event']:
                tokens = doc.geneSentTokens(link.sour, link.targ)
                link.feat_inputs['word_sdp'] = [tok.content for tok in tokens]
                link.feat_inputs['char_sdp'] = [ list(tok.content.lower()) for tok in tokens]
                link.feat_inputs['sour_dist_sdp'] = getMentionDist(tokens, link.sour)
                link.feat_inputs['targ_dist_sdp'] = getMentionDist(tokens, link.targ)
                link.feat_inputs['sour_word_tok'] = link.sour.content.split()
                link.feat_inputs['targ_word_tok'] = link.targ.content.split()
                link.feat_inputs['sour_dist_tok'], link.feat_inputs['targ_dist_tok']  = getEndPosition(link.sour, link.targ)
            elif link_type in ['Event-DCT']:
                tokens = doc.geneSentOfMention(link.sour)
                link.feat_inputs['word_sdp'] = [tok.content for tok in tokens]
                link.feat_inputs['char_sdp'] = [list(tok.content.lower()) for tok in tokens]
                link.feat_inputs['sour_dist_sdp'] = getMentionDist(tokens, link.sour)
                link.feat_inputs['sour_word_tok'] = link.sour.content.split()

    word_vocab = doc2fvocab(doc_dic, 'word_sdp', link_type)
    char_vocab = wvocab2cvocab(word_vocab)
    char_idx = vocab2idx(char_vocab, feat_idx={'zeropadding': 0})
    max_char_len = max([len(word) for word in word_vocab])

    # create word index map or pre-trained embedding
    if pretrained_file and os.path.isfile(os.path.join(os.getenv("HOME"), pretrained_file)):
        pre_model, word_idx = load_pre(os.path.join(os.getenv("HOME"), pretrained_file))
    else:
        pre_model = None
        word_idx = vocab2idx(word_vocab, feat_idx={'zeropadding': 0})
        # word_idx = feat2idx(doc_dic, 'token_sdp', link_type, feat_idx={'zeropadding': 0})

    # create feat index map
    if link_type in ['Event-Timex', 'Event-Event']:
        sour_dist_vocab = doc2fvocab(doc_dic, 'sour_dist_sdp', link_type)
        sour_dist_idx = vocab2idx(sour_dist_vocab, feat_idx={'zeropadding': 0})
        targ_dist_vocab = doc2fvocab(doc_dic, 'targ_dist_sdp', link_type)
        dist_idx = vocab2idx(targ_dist_vocab, feat_idx=sour_dist_idx)
        max_tok_len = max(max_length(doc_dic, 'sour_word_tok', link_type), max_length(doc_dic, 'targ_word_tok', link_type))
    elif link_type in ['Event-DCT']:
        dist_vocab = doc2fvocab(doc_dic, 'sour_dist_sdp', link_type)
        dist_idx = vocab2idx(dist_vocab, feat_idx={'zeropadding': 0})
        max_tok_len = max_length(doc_dic, 'sour_word_tok', link_type)
    rel_idx = rel2idx(doc_dic, link_type)
    max_sdp_len = max_length(doc_dic, 'word_sdp', link_type)
    print('word vocab size: %i, char vocab size: %i, dist size: %i, relation size: %i\n' \
          'max word len of seq: %i , max char len of word: %i, , max word len of mention: %i' %  (len(word_idx),
                                                                                                 len(char_idx),
                                                                                                 len(dist_idx),
                                                                                                 len(rel_idx),
                                                                                                 max_sdp_len,
                                                                                                 max_char_len,
                                                                                                 max_tok_len))
    return doc_dic, word_idx, char_idx, dist_idx, rel_idx, \
           max_sdp_len, max_tok_len, max_char_len, pre_model


def distrib_labels(doc_dic):

    link_types = ['Event-DCT', 'Event-Timex']

    # print label distribution for each link type

    for link_type in link_types:
        label_dic = defaultdict(dict)
        for doc_id, doc in doc_dic.items():
            for link in doc.get_links_by_type(link_type):
                if link.rel is None:
                    continue
                label_dic[link_type][link.rel] = label_dic[link_type].setdefault(link.rel, 0) + 1
        link_count = 0
        for doc in doc_dic.values():
            for link in doc.get_links_by_type(link_type):
                if link.rel is None:
                    continue
                else:
                    link_count += 1
        all_count = sum([value for value in label_dic[link_type].values()])
        print("[Statistics]Number of %s" % link_type, link_count)
        print(label_dic)
        for label, v in sorted(label_dic[link_type].items(), key=lambda d:d[1], reverse=True):
            count = v
            print("label %s, num %i, rate %.2f%%" % (label, count, count * 100 / link_count))


def distrib_labels2(doc_dic):

    link_types = ['Event-DCT', 'Event-Timex']

    # print label distribution for each link type

    for link_type in link_types:
        label_dic = defaultdict(dict)
        for doc_id, doc in doc_dic.items():
            for link in doc.get_links_by_type(link_type):
                if link.rel is None:
                    continue
                for i in range(len(link.rel)):
                    label = 'u%i-%s' % (i, link.rel[i])
                    label_dic[link_type][label] = label_dic[link_type].setdefault(label, 0) + 1
        link_count = 0
        for doc in doc_dic.values():
            for link in doc.get_links_by_type(link_type):
                if link.rel is None:
                    continue
                else:
                    link_count += 1
        all_count = sum([value for value in label_dic[link_type].values()])
        print("[Statistics]Number of %s" % link_type, link_count)
        print(label_dic)
        for label in sorted(label_dic[link_type].keys()):
            count = label_dic[link_type][label]
            print("label %s, num %i, rate %.2f%%" % (label, count, count * 100 / link_count))


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


def split_train_doc(doc_dic, train_ratio=0.8, seed=13):

    train_dic, val_dic = {}, {}

    random.seed(seed)

    for doc_id, doc in doc_dic.items():
        if random.random() < train_ratio:
            train_dic[doc_id] = doc
        else:
            val_dic[doc_id] = doc

    print("Split train/val doc data: train %i, val %i" % (len(train_dic), len(val_dic)))
    print("Event number: train %i, val %i" % (sum([len(doc.events) for doc in train_dic.values()]),
                                              sum([len(doc.events) for doc in val_dic.values()])))

    return train_dic, val_dic


def split_train_doc_pkl(pkl_file, train_pkl, val_pkl, train_ratio=0.9, seed=13):

    doc_dic = load_from_pickle(pkl_file)

    train_dic, val_dic = {}, {}

    random.seed(seed)

    for doc_id, doc in doc_dic.items():
        if random.random() < train_ratio:
            train_dic[doc_id] = doc
        else:
            val_dic[doc_id] = doc

    print("Split train/val doc data: train %i, val %i" % (len(train_dic), len(val_dic)))
    print("Event number: train %i, val %i" % (sum([len(doc.events) for doc in train_dic.values()]),
                                              sum([len(doc.events) for doc in val_dic.values()])))

    save_to_pickle(train_dic, train_pkl)
    save_to_pickle(val_dic, val_pkl)


def split_full_doc(doc_pickle, data_split=(8, 1, 1), seed=13):

    doc_dic = load_from_pickle(pickle_file=doc_pickle)

    train_dic, val_dic, test_dic = {}, {}, {}

    random.seed(seed)

    train_interval = data_split[0] / sum(data_split)
    val_interval = (data_split[0] + data_split[1]) / sum(data_split)

    for doc_id, doc in doc_dic.items():
        random_num = random.random()
        if random_num <= train_interval:
            train_dic[doc_id] = doc
        elif train_interval < random_num <= val_interval:
            val_dic[doc_id] = doc
        else:
            test_dic[doc_id] = doc

    print("Split full doc data: train %i, val %i, test %i" % (
        len(train_dic),
        len(val_dic),
        len(test_dic)
    ))

    print("Event number: train %i, val %i, test %i" % (
        sum([len(doc.events) for doc in train_dic.values()]),
        sum([len(doc.events) for doc in val_dic.values()]),
        sum([len(doc.events) for doc in test_dic.values()])
    ))

    TBD_doc = {key: value for key, value in train_dic.items() if key.startswith('DNS')}
    AQ_doc = {key: value for key, value in train_dic.items() if key.startswith('AQA')}
    TB_doc = {key: value for key, value in train_dic.items() if key.startswith('S-ALL')}

    return TBD_doc, AQ_doc, TB_doc, val_dic, test_dic


def split_full_doc_pkl(pkl_file, train_pkl, val_pkl, test_pkl, data_split=(3, 1, 1), seed=13):

    doc_dic = load_from_pickle(pkl_file)

    train_dic, val_dic, test_dic = {}, {}, {}

    random.seed(seed)

    train_interval = data_split[0] / sum(data_split)
    val_interval = (data_split[0] + data_split[1]) / sum(data_split)

    for doc_id, doc in doc_dic.items():
        random_num = random.random()
        if random_num <= train_interval:
            train_dic[doc_id] = doc
        elif train_interval < random_num <= val_interval:
            val_dic[doc_id] = doc
        else:
            test_dic[doc_id] = doc

    print("Split train/val doc data: train %i, val %i, test %i" % (len(train_dic),
                                                                   len(val_dic),
                                                                   len(test_dic)))

    print("Event number: train %i, val %i, test %i" % (
        sum([len(doc.events) for doc in train_dic.values()]),
        sum([len(doc.events) for doc in val_dic.values()]),
        sum([len(doc.events) for doc in test_dic.values()])
    ))

    save_to_pickle(train_dic, train_pkl)
    save_to_pickle(val_dic, val_pkl)
    save_to_pickle(test_dic, test_pkl)


def prepare_full_doc_pkl(anchorml_dir, all_pkl, train_pkl, val_pkl, test_pkl):

    anchorML_to_doc_pickle(anchorml_dir, all_pkl)
    split_full_doc_pkl(all_pkl, train_pkl, val_pkl, test_pkl, data_split=(4, 1, 1), seed=1337)


def prepare_TBD_doc_pkl(TBD_dir, train_pkl, val_pkl, test_pkl):

    anchorML_to_doc_pickle(os.path.join(TBD_dir, 'TBD_Train'), train_pkl)
    anchorML_to_doc_pickle(os.path.join(TBD_dir, 'TBD_Val'), val_pkl)
    anchorML_to_doc_pickle(os.path.join(TBD_dir, 'TBD_Test'), test_pkl)


def main2():

    full_anchorml_dir = os.path.join(os.environ['HOME'], 'Resources/timex/AnchorData/20190222/ALL')
    full_pickle = "data/eventime/20190222/doc_data/FULL.pkl"

    TBD_doc, AQ_doc, TB_doc, val_doc, test_doc = split_full_doc(
        full_pickle,
        data_split=(8, 1, 1),
        seed=303
    )
    print(len(TBD_doc))
    save_to_pickle(TBD_doc, "data/eventime/%s/doc_data/%s.pkl" % ('20190222', 'F_TBD'))
    print(len(AQ_doc))
    save_to_pickle(AQ_doc, "data/eventime/%s/doc_data/%s.pkl" % ('20190222', 'F_AQ'))
    print(len(TB_doc))
    save_to_pickle(TB_doc, "data/eventime/%s/doc_data/%s.pkl" % ('20190222', 'F_TB'))
    print(len(val_doc))
    save_to_pickle(val_doc, "data/eventime/%s/doc_data/%s.pkl" % ('20190222', 'F_VAL'))
    print(len(test_doc))
    save_to_pickle(test_doc, "data/eventime/%s/doc_data/%s.pkl" % ('20190222', 'F_TEST'))
    print(len(TBD_doc) + len(AQ_doc) + len(TB_doc) + len(val_doc) + len(test_doc))


def main():

    update_label = 3  # 1: {'0', '1'}, 3: {'A', 'B', 'S', 'V'}
    addSEP = False
    reverse_rel = False
    home = os.environ['HOME']

    is_reset_doc = False

    is_generate_date = True

    data_dir = '20190222'  # '20190202', '20190222' or 'TBD'

    all_datasets = ['TB', 'AQ', 'TBD_TRAIN', 'TBD_VAL', 'TBD_TEST', 'TE3_TEST']

    train_datasets = ['TBD_TRAIN']

    val_datasets = ['TBD_VAL']

    test_datasets = ['TBD_TEST']

    if data_dir == '20190222':

        if is_reset_doc:
            for dataset in all_datasets:
                anchorml_dir = os.path.join(home, "Resources/timex/AnchorData/20190222/%s" % dataset)
                pickle_doc = "data/eventime/%s/doc_data/%s.pkl" % (data_dir, dataset)
                anchorML_to_doc_pickle(anchorml_dir, pickle_doc)

    if is_generate_date:

        dataset_flag = 'T:%s:V:%s:T:%s' % (
            '-'.join(train_datasets),
            '-'.join(val_datasets),
            '-'.join(test_datasets)
        )

        tensor_dir = 'data/eventime/%s/%s' % (data_dir, dataset_flag)

        if not os.path.exists(tensor_dir):
            os.makedirs(tensor_dir)

        train_doc, val_doc, test_doc = {}, {}, {}

        for dataset in train_datasets:
            pickle_doc = "data/eventime/%s/doc_data/%s.pkl" % (data_dir, dataset)
            data_doc = load_from_pickle(pickle_file=pickle_doc)
            train_doc = {**train_doc, **data_doc}

        for dataset in test_datasets:
            pickle_doc = "data/eventime/%s/doc_data/%s.pkl" % (data_dir, dataset)
            data_doc = load_from_pickle(pickle_file=pickle_doc)
            test_doc = {**test_doc, **data_doc}

        if val_datasets:
            for dataset in val_datasets:
                pickle_doc = "data/eventime/%s/doc_data/%s.pkl" % (data_dir, dataset)
                data_doc = load_from_pickle(pickle_file=pickle_doc)
                val_doc = {**val_doc, **data_doc}
        else:
            train_doc, val_doc = split_train_doc(train_doc, train_ratio=0.8, seed=13)

        train_doc_dic = preprocess_doc(train_doc, oper=update_label, sent_win=5, reverse_rel=reverse_rel)

        val_doc_dic = preprocess_doc(val_doc, oper=update_label, sent_win=2, reverse_rel=reverse_rel)

        test_doc_dic = preprocess_doc(test_doc, oper=update_label, sent_win=1, reverse_rel=reverse_rel)

        test_gold = prepare_gold(test_doc_dic)

        print(len(test_gold))

        ed_train_fields, et_train_fields, _, _ = prepare_feats(train_doc_dic, addSEP=addSEP)

        ed_val_fields, et_val_fields, _, _ = prepare_feats(val_doc_dic, addSEP=addSEP)

        ed_test_fields, et_test_fields, test_indices, test_targ = prepare_feats(test_doc_dic, addSEP=addSEP)

        print(len(test_indices[0]), len(test_indices[1]))
        print(ed_train_fields.keys(), et_train_fields.keys())

        save_to_pickle(
            (test_gold, test_indices[0], test_indices[1], test_targ[0], test_targ[1]),
            'data/eventime/%s/%s/test_gold.pkl' % (data_dir, dataset_flag)
        )

        for link_type in ['Event-DCT', 'Event-Timex']:

            train_instances = DataInstance(locals()['%s_train_fields' % ('ed' if link_type == 'Event-DCT' else 'et')])
            val_instances = DataInstance(locals()['%s_val_fields' % ('ed' if link_type == 'Event-DCT' else 'et')])
            test_instances = DataInstance(locals()['%s_test_fields' % ('ed' if link_type == 'Event-DCT' else 'et')])

            min_count = defaultdict(lambda: 0)
            max_vocab_size = defaultdict(lambda: 0)
            zero_unk = defaultdict(lambda: True)
            zero_unk['labels'] = False

            vocab, padding_lengths = get_vocab_and_lengths(
                train_instances,
                val_instances,
                test_instances,
                min_count,
                max_vocab_size,
                zero_unk
            )

            print(len(vocab.get_token_to_index()['tokens']))
            print(vocab.get_token_to_index()['labels'])

            embed_file = os.path.join(home, "Resources/embed/giga-aacw.d200.bin")
            embed_pickle_file = "data/eventime/%s/%s/giga.d200.%s.l%i.embed" % (
                data_dir,
                dataset_flag,
                link_type,
                update_label
            )

            slim_word_embed(vocab.get_token_to_index()['tokens'], embed_file, embed_pickle_file)

            train_instances.index_field(vocab)
            val_instances.index_field(vocab)
            test_instances.index_field(vocab)

            train_tensor_dataset = train_instances.as_tensor_dict(padding_lengths)

            val_tensor_dataset = val_instances.as_tensor_dict(padding_lengths)

            test_tensor_dataset = test_instances.as_tensor_dict(padding_lengths)

            print(train_tensor_dataset.keys())
            print(train_tensor_dataset['tokens'].shape,
                  val_tensor_dataset['tokens'].shape,
                  test_tensor_dataset['tokens'].shape)

            save_to_pickle(train_tensor_dataset, 'data/eventime/%s/%s/train_t_%s_l%i.pkl' % (
                data_dir,
                dataset_flag,
                link_type,
                update_label
            ))

            save_to_pickle(val_tensor_dataset, 'data/eventime/%s/%s/val_t_%s_l%i.pkl' % (
                data_dir,
                dataset_flag,
                link_type,
                update_label
            ))

            save_to_pickle(test_tensor_dataset, 'data/eventime/%s/%s/test_t_%s_l%i.pkl' % (
                data_dir,
                dataset_flag,
                link_type,
                update_label
            ))

            save_to_pickle((vocab, padding_lengths), 'data/eventime/%s/%s/glob_info_%s_l%i.pkl' % (
                data_dir,
                dataset_flag,
                link_type,
                update_label
            ))


if __name__ == '__main__':
    main()


# import unittest
# # unittest.main(warnings='ignore')
#
#
# class TestTempData(unittest.TestCase):
#
#     def test_prepare_global(self):
#         link_type = 'Event-Timex'
#         pkl_file = "data/0531_%s.pkl" % (link_type)
#         doc_dic, word_idx, char_idx, dist_idx, rel_idx, \
#         max_sdp_len, max_token_len, max_char_len, pre_model = prepare_global(pkl_file, None)
#
#     def test_prepare_feat_dataset(self):
#         link_type = 'Event-Timex'
#         pkl_file = "data/0531.pkl"
#         feat_types = ['word_sdp',
#                     # 'char_sdp',
#                     'sour_dist_sdp',
#                     'targ_dist_sdp',
#                     'sour_word_tok',
#                     'targ_word_tok',
#                     'sour_dist_tok',
#                     'targ_dist_tok']
#         doc_dic, word_idx, char_idx, dist_idx, rel_idx, \
#         max_sdp_len, max_tok_len, max_char_len, pre_model = prepare_global(pkl_file, None, link_type)
#         train_inputs, train_target = prepare_feats_dataset(doc_dic, TA_TEST, word_idx, char_idx, dist_idx, rel_idx, max_sdp_len, max_tok_len, max_char_len, link_type, feat_types=feat_types)
#         print([ (feat_type, feat.shape) for feat, feat_type in zip(train_inputs, feat_types)], train_target.shape)
#
#     def test_anchorML2doc(self):
#         anchorml = "/Users/fei-c/Resources/timex/AnchorData/all_20190202/train"
#         pkl_file = "data/20190202_train.pkl"
#         AnchorML2doc(anchorml, pkl_file)
#
#     def test_anchor_file2doc(self):
#         sent_win = 0
#         oper = True
#         timeml_dir = os.path.join(os.path.dirname(__file__), "data/Timebank")
#         anchor_file = os.path.join(os.path.dirname(__file__), "data/event-times_normalized.tab")
#         pkl_file = os.path.join(os.path.dirname(__file__),
#                                 "data/unittest-%s-%s_w%i_%s.pkl" % (timeml_dir.split('/')[-1],
#                                                                     anchor_file.split('/')[-1],
#                                                                     sent_win,
#                                                                     'oper' if oper else 'order'))
#         # anchor_file2doc(timeml_dir, anchor_file, pkl_file, sent_win, oper=True)
#
#         doc_dic = load_doc(pkl_file)
#
#         # doc = doc_dic['ABC19980304.1830.1636']
#
#         e_d = defaultdict(lambda: 0)
#         e_t = defaultdict(lambda: 0)
#
#         for doc in doc_dic.values():
#             for event_id, event in doc.events.items():
#                 # print(event_id, event.value)
#                 for link in doc.getTlinkListByMention(event_id):
#                     if not link.sour.tanchor:
#                         continue
#                     if link.link_type == 'Event-DCT':
#                         e_d[link.rel] += 1
#                     if link.link_type == 'Event-Timex':
#                         e_t[link.rel] += 1
#
#         print(len(e_d.keys()), len(e_t.keys()))
#         sum_e_d = sum(e_d.values())
#         sum_e_t = sum(e_t.values())
#         print(sum_e_d, sum_e_t)
#         print()
#
#         print("Event-DCT:")
#         for key, value in sorted(e_d.items(), key=lambda kv: kv[1], reverse=True):
#             print(key, "%.2f" % (value/sum_e_d))
#         print()
#
#         print("Event-Timex:")
#         for key, value in sorted(e_t.items(), key=lambda kv: kv[1], reverse=True):
#             print(key, "%.2f" % (value/sum_e_t))
#
#     def test_loadTimeML2str(self):
#         timeml_file = "/Users/fei-c/Resources/timex/20180919/kd/AQA024_APW199980817.1193.tml"
#         print(loadTimeML2str(timeml_file))
#         writeTimeML2txt(timeml_file)
#
#     def test_writeCommonTimeML2txt(self):
#         writeCommonTimeML2txt("/Users/fei-c/Resources/timex/20180919/kd", "/Users/fei-c/Resources/timex/20180919/td", "data/ref")
#
#     def test_prepareGlobalSDP(self):
#
#         sent_win = 1
#         link_type = 'Event-DCT'
#         timeml_dir = os.path.join(os.path.dirname(__file__), "data/Timebank")
#         anchor_file = os.path.join(os.path.dirname(__file__), "data/event-times_normalized.tab")
#         pkl_file = os.path.join(os.path.dirname(__file__), "data/unittest-%s-%s_w%i.pkl" % (timeml_dir.split('/')[-1],
#                                                                                             anchor_file.split('/')[-1],
#                                                                                             sent_win))
#
#         # anchor_file2doc(timeml_dir, anchor_file, pkl_file, sent_win, oper=False)
#
#         doc_dic, word_idx, char_idx, pos_idx, dep_idx, rel_idx, \
#         max_sdp_len, max_mention_len, max_word_len = prepareGlobalSDP(pkl_file, link_types=['Event-DCT', 'Event-Timex'])
#
#         pretrained_file = "Resources/embed/giga-aacw.d200.bin"
#         pre_lookup_table, word_idx = readPretrainedEmbedding(pretrained_file)
#         print(pre_lookup_table.shape)
#         '''
#         print train/dev/test feats shape
#         '''
#         train_feats, train_target = feat2tensorSDP(doc_dic, TBD_TRAIN, word_idx, char_idx, pos_idx, dep_idx, rel_idx,
#                                                    max_sdp_len, max_mention_len, max_word_len, link_type)
#         print(len(train_feats))
#         print(train_feats['sour_word_sdp'].shape)
#         print(train_feats['sour_char_sdp'].shape)
#         print(train_feats['sour_word_tok'].shape)
#
#         dev_feats, dev_target = feat2tensorSDP(doc_dic, TBD_DEV, word_idx, char_idx, pos_idx, dep_idx, rel_idx,
#                                                max_sdp_len, max_mention_len, max_word_len, link_type)
#         print(len(dev_feats))
#         print(dev_feats['sour_word_sdp'].shape)
#         print(dev_feats['sour_char_sdp'].shape)
#         print(train_feats['sour_word_tok'].shape)
#
#         test_feats, test_target = feat2tensorSDP(doc_dic, TBD_TEST, word_idx, char_idx, pos_idx, dep_idx, rel_idx,
#                                                  max_sdp_len, max_mention_len, max_word_len, link_type)
#         print(len(test_feats))
#         print(test_feats['sour_word_sdp'].shape)
#         print(test_feats['sour_char_sdp'].shape)
#         print(train_feats['sour_word_tok'].shape)