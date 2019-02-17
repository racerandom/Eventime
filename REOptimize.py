# coding=utf-8

import warnings
import logging
import time
import random
from collections import defaultdict

import torch
import torch.utils.data as Data

import ModuleOptim
import TempEval
import TempModule2
import TempUtils

warnings.simplefilter("ignore", UserWarning)


logger = TempUtils.setup_stream_logger(
    'REOptimize',
    level=logging.DEBUG
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

seed = 1
random.seed(seed)

torch_seed = 1337
torch.manual_seed(torch_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def get_checkpoint_file(checkpoint_base, monitor, score):
    return "%s_%s_%f.pth" % (checkpoint_base,
                             monitor,
                             score)


def model_instance_ET(word_size, dist_size, targ2ix,
                      max_sent_len, pre_embed, **params):

    model = getattr(TempModule2, params['classification_model'])(
        word_size, dist_size, targ2ix,
        max_sent_len, pre_embed, **params
    ).to(device=device)

    optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=params['lr'],
                                     weight_decay=params['weight_decay'])
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                              lr=params['lr'],
    #                              weight_decay=params['weight_decay'])

    logger.debug(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug('* %s' % name)
        else:
            logger.debug('%s' % name)

    logger.debug("Parameters: %i" % ModuleOptim.count_parameters(model))

    return model, optimizer


def data_load_ET():

    link_type = 'Event-Timex'

    train_dataset = TempUtils.load_pickle('data/eventime/120190202_train_tensor_%s.pkl' % link_type)

    val_dataset = TempUtils.load_pickle('data/eventime/120190202_val_tensor_%s.pkl' % link_type)

    test_dataset = TempUtils.load_pickle('data/eventime/120190202_test_tensor_%s.pkl' % link_type)

    word2ix, pretrained_embed = TempUtils.load_pickle('data/eventime/giga.d200.%s.embed' % link_type)

    ldis2ix, targ2ix, max_sent_len = TempUtils.load_pickle('data/eventime/120190202_glob_info_%s.pkl' % link_type)

    return train_dataset, val_dataset, test_dataset, \
           word2ix, ldis2ix, targ2ix, max_sent_len, pretrained_embed


def optimize_model(train_file, test_file, embed_file, param_space,
                   feat_dict, max_evals=10):

    print('device:', device)

    monitor = param_space['monitor'][0]

    checkpoint_base = "models/checkpoint_%s_%i" % (param_space['classification_model'][0],
                                                   int(time.time()))

    # train_dataset, val_dataset, test_dataset, \
    # word2ix, dsdp2ix, targ2ix, max_sent_len, max_sdp_len = data_load_RE(train_file, test_file, embed_file, feat_dict)

    train_dataset, val_dataset, test_dataset, \
    word2ix, ldis2ix, targ2ix, max_sent_len, pretrained_embed = data_load_ET()

    logger.info('Word size %i, ldis size %i, max sentence len %i...' % (
        len(word2ix),
        len(ldis2ix),
        max_sent_len
    ))
    logger.info('Train/Val/Test data size: %i / %i / %i' % (len(train_dataset[-1]),
                                                            len(val_dataset[-1]),
                                                            len(test_dataset[-1])))
    if len(ldis2ix) == 0:
        train_dataset = (train_dataset[0], train_dataset[-1])
        val_dataset = (val_dataset[0], val_dataset[-1])
        test_dataset = (test_dataset[0], test_dataset[-1])

    global_eval_history = defaultdict(list)

    monitor_score_history = global_eval_history['monitor_score']

    params_history = []

    kbest_scores = []

    for eval_i in range(1, max_evals + 1):

        params = {}

        while not params or params in params_history:
            for key, values in param_space.items():
                params[key] = random.choice(values)

        logger.info('[Selected %i Params]: %s' % (eval_i, params))

        model, optimizer = model_instance_ET(len(word2ix), len(ldis2ix), targ2ix,
                                             max_sent_len, pretrained_embed, **params)

        kbest_scores = train_model_ET(
            model, optimizer, kbest_scores,
            train_dataset, val_dataset, test_dataset,
            # targ2ix,
            checkpoint_base,
            **params
        )

        logger.info("Kbest scores: %s" % kbest_scores)

    best_index = 0 if monitor.endswith('loss') else -1

    best_checkpoint_file = get_checkpoint_file(checkpoint_base, monitor, kbest_scores[best_index])

    best_checkpoint = torch.load(best_checkpoint_file,
                                 map_location=lambda storage,
                                 loc: storage)

    # logger.info("test_acc, mean: %.4f, stdev: %.4f" % (mean(test_acc_history), stdev(test_acc_history)))
    logger.info("Final best %s: %.4f" % (monitor,
                                         best_checkpoint['best_score']))
    logger.info("Final best params: %s" % best_checkpoint['params'])

    params = best_checkpoint['params']

    model = getattr(TempModule2, params['classification_model'])(
        len(word2ix), len(ldis2ix), targ2ix,
        max_sent_len, pretrained_embed, **params
    ).to(device)

    test_data_loader = Data.DataLoader(
        dataset=ModuleOptim.CustomizedDatasets(*test_dataset),
        batch_size=128,
        collate_fn=ModuleOptim.collate_fn,
        shuffle=False,
        num_workers=1,
    )

    val_data_loader = Data.DataLoader(
        dataset=ModuleOptim.CustomizedDatasets(*val_dataset),
        batch_size=128,
        collate_fn=ModuleOptim.collate_fn,
        shuffle=False,
        num_workers=1,
    )

    model.load_state_dict(best_checkpoint['state_dict'])

    # loss_func = REModule.ranking_loss if param_space['ranking_loss'][0] else F.nll_loss
    # REEval.batch_eval(model,
    #                   test_data_loader,
    #                   targ2ix,
    #                   loss_func,
    #                   param_space['omit_other'][0],
    #                   report_result=True)
    loss_func = ModuleOptim.multilabel_loss

    val_loss, val_acc = TempEval.batch_eval_ET(model, val_data_loader, loss_func, report=True)

    test_loss, test_acc = TempEval.batch_eval_ET(model, test_data_loader, loss_func, report=True)


def train_model_ET(model, optimizer, kbest_scores,
                   train_data, val_data, test_data,
                 # targ2ix,
                   checkpoint_base, **params):

    monitor = params['monitor']

    multi_loss = ModuleOptim.multilabel_loss

    train_data_loader = Data.DataLoader(
        dataset=ModuleOptim.CustomizedDatasets(*train_data),
        batch_size=params['batch_size'],
        collate_fn=ModuleOptim.collate_fn,
        shuffle=True,
        num_workers=1,
    )

    val_data_loader = Data.DataLoader(
        dataset=ModuleOptim.CustomizedDatasets(*val_data),
        batch_size=params['batch_size'],
        collate_fn=ModuleOptim.collate_fn,
        shuffle=True,
        num_workers=1,
    )

    test_data_loader = Data.DataLoader(
        dataset=ModuleOptim.CustomizedDatasets(*test_data),
        batch_size=params['batch_size'],
        collate_fn=ModuleOptim.collate_fn,
        shuffle=False,
        num_workers=1,
    )

    eval_history = defaultdict(list)

    monitor_score_history = eval_history[params['monitor']]

    patience = params['patience']

    for epoch in range(1, params['epoch_num'] + 1):

        epoch_start_time = time.time()

        epoch_losses = []
        epoch_acces = []

        step_num = len(train_data_loader)

        for step, train_batch in enumerate(train_data_loader):

            start_time = time.time()

            train_batch = ModuleOptim.batch_to_device(train_batch, device)
            train_feats = train_batch[:-1]
            train_targ = train_batch[-1]

            model.train()
            model.zero_grad()

            pred_prob = model(*train_feats)

            train_loss = multi_loss(pred_prob, train_targ)
            # batch_size = pred_prob.shape[0]
            # train_loss = F.nll_loss(pred_prob.view(batch_size * 4, -1), train_targ.view(-1))

            train_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['max_norm'])
            optimizer.step()

            epoch_losses.append(train_loss.item())
            # pred_class = torch.gt(pred_prob, 0.5).long()
            # train_acc = (pred_class == train_targ).sum().item() / float(pred_prob.shape[0] * 8)
            # train_acc = ModuleOptim.calc_acc(pred_class, train_targ)
            train_acc = ModuleOptim.calc_multi_acc(pred_prob, train_targ)
            # print(torch.argmax(pred_prob, dim=-1)[0], train_targ[0])

            epoch_acces.append(train_acc)

            if (step != 0 and step % params['check_interval'] == 0) or step == step_num - 1:

                val_loss, val_acc = TempEval.batch_eval_ET(model, val_data_loader, multi_loss)

                eval_history['val_loss'].append(val_loss)
                eval_history['val_acc'].append(val_acc)

                monitor_score = round(locals()[monitor], 6)

                # global_is_best, global_best_score = ModuleOptim.is_best_score(monitor_score,
                #                                                               global_best_score,
                #                                                               monitor)

                is_kbest, kbest_scores = ModuleOptim.update_kbest_scores(kbest_scores,
                                                                         monitor_score,
                                                                         monitor,
                                                                         kbest=params['kbest_checkpoint'])
                # print(kbest_scores)

                if is_kbest and len(kbest_scores) == params['kbest_checkpoint'] + 1:
                    removed_index = -1 if monitor.endswith('loss') else 0
                    removed_score = kbest_scores.pop(removed_index)
                    ModuleOptim.delete_checkpoint(get_checkpoint_file(checkpoint_base,
                                                                      monitor,
                                                                      removed_score))
                    assert len(kbest_scores) == params['kbest_checkpoint']

                global_save_info = ModuleOptim.save_checkpoint({
                    'params': params,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'monitor': monitor,
                    'best_score': monitor_score,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, is_kbest, get_checkpoint_file(checkpoint_base,
                                                 monitor,
                                                 monitor_score))

                test_loss, test_acc = TempEval.batch_eval_ET(model, test_data_loader, multi_loss)

                logger.debug(
                    'epoch: %2i, step: %4i, time: %4.1fs | '
                    'train loss: %.4f, train acc: %.4f | '
                    'val loss: %.4f, val acc: %.4f | '
                    'test loss: %.4f, test acc: %.4f %s'
                    % (
                        epoch,
                        step,
                        time.time() - start_time,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                        test_loss,
                        test_acc,
                        global_save_info
                    )
                )

        eval_history['epoch_best'].append(
            ModuleOptim.get_best_score(monitor_score_history[-step_num * params['check_interval']:],
                                       monitor)
        )

        logger.info(
            "epoch %i finished in %.2fs, "
            "train loss: %.4f, train acc: %.4f"
            % (
                epoch,
                time.time() - epoch_start_time,
                sum(epoch_losses) / float(len(epoch_losses)),
                sum(epoch_acces) / float(len(epoch_acces)),
               )
        )

        if (patience and
            len(eval_history['epoch_best']) >= patience and
            eval_history['epoch_best'][-patience] == ModuleOptim.get_best_score(eval_history['epoch_best'][-patience:],
                                                                                monitor)):
            print('[Early Stopping] patience reached, stopping...')
            break

    best_local_index = monitor_score_history.index(ModuleOptim.get_best_score(monitor_score_history, params['monitor']))

    return kbest_scores


def main():

    classification_model = 'TempRNN'   # 'baseRNN', 'attnRNN'

    param_space = {
        'classification_model': [classification_model],
        'freeze_mode': [False],
        'sdp_filter_nb': [100],     # SDP representation for each token
        'sdp_kernel_len': [3],
        'sdp_cnn_droprate': [0.3],
        'sdp_fc_dim': [100],
        'sdp_fc_droprate': [0.3],
        'dsdp_dim': [25],
        'dist_dim': [25],
        'input_dropout': [0.3],     # hyper-parameters of neural networks
        'rnn_hidden_dim': [300],
        'rnn_layer': [1],
        'rnn_dropout': [0.3],
        'attn_dropout': [0.3],
        'fc1_hidden_dim': [300],
        'fc1_dropout': [0.5],
        'batch_size': [64],
        'epoch_num': [200],
        'lr': [1e-0],           # hyper-parameters of optimizer
        'weight_decay': [1e-4],
        'max_norm': [4],
        'patience': [10],       # early stopping
        'monitor': ['val_acc'],
        'check_interval': [100],    # checkpoint based on val performance given a step interval
        'kbest_checkpoint': [5],
        'ranking_loss': [False],    # ranking loss for the baseRNN model
        'omit_other': [False],
        'gamma': [2],
        'margin_pos': [2.5],
        'margin_neg': [0.5],
    }

    feat_dict = {
        'PI': True,   # Position Indicator Features for 'baseRNN', 'attnRNN'
        'SDP': False,
        'TSDP': False,
        'DSDP': False
    }

    feat_suffix = ''
    for k, v in feat_dict.items():
        if v:
            feat_suffix += '.%s' % k

    train_file = "data/train%s.pkl" % feat_suffix
    test_file = "data/test%s.pkl" % feat_suffix
    embed_file = "data/glove.100d.embed"

    optimize_model(train_file, test_file, embed_file, param_space, feat_dict, max_evals=1)


if __name__ == '__main__':
    main()
