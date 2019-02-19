# coding=utf-8
import warnings
import os
from bisect import bisect

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F

warnings.simplefilter("ignore", UserWarning)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_best_score(score, best_score, monitor):
    if not best_score:
        is_best = True
        best_score = score
    else:
        is_best = bool(score < best_score) if monitor.endswith('_loss') else bool(score > best_score)
        best_score = score if is_best else best_score
    return is_best, best_score


def save_checkpoint(state, is_best, filename):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        torch.save(state, filename)  # save checkpoint
        return "=> Saving a new best"
    else:
        return ""


def multilabel_loss(pred_prob, targets, average_size=False):

    batch_size, pos_size = targets.shape
    loss = F.nll_loss(pred_prob[:, 0, :], targets[:, 0])
    for i in range(1, pred_prob.shape[1]):
        loss += F.nll_loss(pred_prob[:, i, :], targets[:, i])
    if average_size:
        return loss / pos_size
    else:
        return loss


def calc_multi_acc(pred_prob, targets):
    pred_class = torch.argmax(pred_prob, dim=-1)
    batch_size = pred_class.shape[0]
    correct = 0
    for i in range(batch_size):
        if torch.equal(pred_class[i], targets[i]):
            correct += 1
    return correct / batch_size


def pred_ix_to_label(pred_ix, ix2targ):
    return [''.join([ix2targ[i.item()] for i in ix]) for ix in pred_ix]





def update_kbest_scores(kbest_scores, new_score, monitor, kbest=5):

    def is_kbest(score, kbest):
        if monitor.endswith('loss'):
            if score < max(kbest):
                return True
            else:
                return False
        else:
            if score > min(kbest):
                return True
            else:
                return False

    if len(kbest_scores) < kbest:
        if new_score not in kbest_scores:
            new_index = bisect(kbest_scores, new_score)
            kbest_scores.insert(new_index, new_score)
            is_kbest = True
            return is_kbest, kbest_scores
        else:
            is_kbest = False
            return is_kbest, kbest_scores
    else:
        assert len(kbest_scores) == kbest
        if is_kbest(new_score, kbest_scores) and new_score not in kbest_scores:
            new_index = bisect(kbest_scores, new_score)
            kbest_scores.insert(new_index, new_score)
            is_kbest = True
            return is_kbest, kbest_scores    # len is kbest + 1, ready to pop the last for delete checkpoint
        else:
            is_kbest = False
            return is_kbest, kbest_scores


def calc_acc(pred_class, targs):
    correct = 0
    for i in range(pred_class.shape[0]):
        if torch.equal(pred_class[i], targs[i]):
            correct += 1
    return correct / pred_class.shape[0]


def delete_checkpoint(filename):
    if os.path.isfile(filename):
        os.remove(filename)
    else:
        raise Exception('[ERROR] Attempt to delete an unexisting checkpoint %s...' % filename)


def get_best_score(scores, monitor):
    if not scores:
        return None
    else:
        if monitor.endswith('acc'):
            return max(scores)
        elif monitor.endswith('loss'):
            return min(scores)
        elif monitor.endswith('f1'):
            return max(scores)
        else:
            raise Exception('[ERROR] Unknown monitor mode...')


def batch_to_device(data, device):
    """
    copy input list into device for GPU computation
    :param data:
    :param device:
    :return:
    """
    device_inputs = []
    for d in data:
        device_inputs.append(d.to(device=device))
    return device_inputs


def copyData2device(data, device):
    feat_dict, target = data
    feat_types = list(feat_dict.keys())
    feat_list = batch_to_device(list(feat_dict.values()), device)
    target = target.to(device=device)
    return dict(zip(feat_types, feat_list)), target


class CustomizedDatasets(Data.Dataset):
    """Dataset wrapping list.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    def __init__(self, *listData):
        assert all(len(listData[0]) == len(data) for data in listData)
        self.listData = listData

    def __getitem__(self, index):
        return tuple(data[index] for data in self.listData)

    def __len__(self):
        return len(self.listData[0])


def collate_fn(batch_data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (words: tensor, entity1_ids: list, entity2_ids: list, targets:tensor).
            - words: torch tensor of shape (batch, max_len, word_dim).
            - entity1_ids:
            - entity2_ids
            - targets: torch tensor of shape (batch); variable length.
    """
    feat_num = len(batch_data[0])

    out_list = []

    for i in range(feat_num):
        data_feat = [data[i] for data in batch_data]
        if isinstance(batch_data[0][i], torch.Tensor):
            out_list.append(torch.stack(data_feat))
        elif isinstance(batch_data[0][i], list):
            out_list.append(data_feat)
        elif isinstance(batch_data[0][i], np.ndarray):
            out_list.append(np.asarray(data_feat))
        else:
            raise Exception("[ERROR] Unknown data type in 'collate_fn'...")

    return out_list

