# coding=utf-8
import logging

from sklearn.metrics import classification_report

import torch
import ModuleOptim
import TempUtils


logger = logging.getLogger('REOptimize')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")


def batch_eval_ET(model, data_loader, loss_func, targ2ix, report=False):

    model.eval()
    with torch.no_grad():

        pred_prob, targ = torch.FloatTensor().to(device), torch.LongTensor().to(device)

        for batch in data_loader:
            batch = ModuleOptim.batch_to_device(batch, device)
            batch_feats = batch[:-1]
            batch_targ = batch[-1]

            batch_prob = model(*batch_feats)

            pred_prob = torch.cat((pred_prob, batch_prob), dim=0)
            targ = torch.cat((targ, batch_targ), dim=0)

        assert pred_prob.shape[0] == targ.shape[0]

    loss = loss_func(pred_prob, targ)

    acc = ModuleOptim.calc_multi_acc(pred_prob, targ)

    pred_ix = torch.argmax(pred_prob, dim=-1)

    ix2targ = {v: k for k, v in targ2ix.items()}

    pred_label = ModuleOptim.pred_ix_to_label(pred_ix, ix2targ)
    targ_label = ModuleOptim.pred_ix_to_label(targ, ix2targ)

    if report:
        # pred = [''.join(str(i.item()) for i in t) for t in pred_ix]
        # targ = [''.join(str(i.item()) for i in t) for t in targ]
        label2ix = TempUtils.label_to_ix(pred_label + targ_label)
        pred_index = [label2ix[t] for t in pred_label]
        targ_index = [label2ix[t] for t in targ_label]

        print(classification_report(targ_index,
                              pred_index,
                              target_names=[k for k, v in sorted(label2ix.items(), key=lambda d: d[1])]
                              ))

    return loss.item(), acc, pred_label



