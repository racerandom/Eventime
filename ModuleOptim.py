# coding=utf-8
import warnings
warnings.simplefilter("ignore", UserWarning)

import torch.utils.data as Data
import torch
from TempData import *
from sklearn.metrics import classification_report
from statistics import mean, median, variance, stdev

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print('device:', device)

seed = 2
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


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


def copyData2device(data, device):
    feat_dict, target = data
    feat_types = list(feat_dict.keys())
    feat_list = batch_to_device(list(feat_dict.values()), device)
    target = target.to(device=device)
    return dict(zip(feat_types, feat_list)), target

