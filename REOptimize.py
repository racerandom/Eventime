# coding=utf-8

import warnings
import logging
import time
import random
from collections import defaultdict
from typing import *

import torch
import torch.utils.data as Data
from torch import nn

import ModuleOptim
import TempEval
import TempModule2
import TempUtils
from TempData import Vocabulary

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


def prepare_embedding_dict(vocab: Vocabulary,
                           pretrained_embed,
                           params: Dict):

    embedding_dict = dict()

    embedding_dict['tokens'] = TempUtils.pre_to_embed(
        pretrained_embed,
        freeze_mode=params['freeze_mode']
    )
    if 'event_dist' in vocab.get_token_to_index():
        embedding_dict['event_dist'] = nn.Embedding(
            len(vocab.get_token_to_index()['event_dist']),
            params['dist_dim']
        )
    elif 'timex_dist' in vocab.get_token_to_index():
        embedding_dict['timex_dist'] = nn.Embedding(
            len(vocab.get_token_to_index()['timex_dist']),
            params['dist_dim']
        )
    elif 'event_dsdp' in vocab.get_token_to_index():
        embedding_dict['event_dsdp'] = nn.Embedding(
            len(vocab.get_token_to_index()['event_dsdp']),
            params['dsdp_dim']
        )
    elif 'timex_dsdp' in vocab.get_token_to_index():
        embedding_dict['timex_dsdp'] = nn.Embedding(
            len(vocab.get_token_to_index()['timex_dsdp']),
            params['dsdp_dim']
        )
    return embedding_dict


def prepare_model_components(link_type, vocab, params):

    model_in_dim = params['word_dim'] + params['dist_dim'] + params['dsdp_dim']

    model = getattr(TempModule2, params['classification_model'])(
        model_in_dim,
        params['rnn_hidden_dim'],
        num_layer=params['rnn_layer'],
        out_droprate=params['rnn_dropout']
    )

    attn_layer = None
    if params['use_attn']:
        attn_in_dim = (2 * model_in_dim) if link_type == 'Event-DCT' else (3 * model_in_dim)
        attn_layer = TempModule2.EntityAttn(attn_in_dim,
                                            params['attn_fc_dim'],
                                            params['attn_fc_drop'],
                                            params['attn_out_drop'])

    out_layer = TempModule2.OutLayer(
        params['rnn_hidden_dim'],
        params['fc1_hidden_dim'],
        params['fc1_dropout'],
        vocab
    )

    return model, attn_layer, out_layer


def optimize_model(link_type,
                   train_pkl, val_pkl, test_pkl,
                   info_pkl, embed_pkl,
                   pred_pkl, targ_pkl,
                   param_space, max_evals=10):

    print('device:', device)

    monitor = param_space['monitor'][0]

    checkpoint_base = "models/checkpoint_%s_%s_%i" % (
        link_type,
        param_space['classification_model'][0],
        int(time.time())
    )

    train_dataset = TempUtils.load_from_pickle(train_pkl)

    val_dataset = TempUtils.load_from_pickle(val_pkl)

    test_dataset = TempUtils.load_from_pickle(test_pkl)

    pretrained_weights = TempUtils.load_from_pickle(embed_pkl)

    param_space['word_dim'] = [pretrained_weights.shape[1]]

    vocab, padding_lengths = TempUtils.load_from_pickle(info_pkl)

    logger.info('Word size %i, event_dist size %i, max sentence len %i...' % (
        len(vocab.get_token_to_index()['tokens']),
        0 if 'dists' not in vocab.get_token_to_index() else len(vocab.get_token_to_index()['event_dist']),
        max(padding_lengths['tokens'])
    ))
    logger.info('Train/Val/Test data size: %s / %s / %s' % (train_dataset['tokens'].shape,
                                                            val_dataset['tokens'].shape,
                                                            test_dataset['tokens'].shape))

    # fix-ordered val/test dataloader
    test_data_loader = Data.DataLoader(
        dataset=ModuleOptim.DictDatasets(
            ModuleOptim.batch_to_device(test_dataset, device)
        ),
        batch_size=128,
        collate_fn=ModuleOptim.dict_collate_fn,
        num_workers=1,
    )

    val_data_loader = Data.DataLoader(
        dataset=ModuleOptim.DictDatasets(
            ModuleOptim.batch_to_device(val_dataset, device)
        ),
        batch_size=128,
        collate_fn=ModuleOptim.dict_collate_fn,
        num_workers=1,
    )

    params_history = []

    kbest_scores = []

    loss_func = ModuleOptim.multilabel_loss
    acc_func = ModuleOptim.calc_element_acc

    for eval_i in range(1, max_evals + 1):

        params = dict()

        while not params or params in params_history:
            for key, values in param_space.items():
                params[key] = random.choice(values)

        logger.info('[Selected %i Params]: %s' % (eval_i, params))

        # preparing the classifier components: embedding layer, model

        embedding_dict = prepare_embedding_dict(vocab, pretrained_weights, params)

        model, attn_layer, out_layer = prepare_model_components(link_type, vocab, params)

        classifier = TempModule2.TempClassifier(
            embedding_dict,
            model,
            attn_layer,
            out_layer
        ).to(device)

        logger.debug(classifier)
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                logger.debug('* %s' % name)
            else:
                logger.debug('%s' % name)

        logger.debug("Parameters: %i" % ModuleOptim.count_parameters(model))

        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, classifier.parameters()),
                                         lr=params['lr'],
                                         weight_decay=params['weight_decay'])

        train_data_loader = Data.DataLoader(
            dataset=ModuleOptim.DictDatasets(
                ModuleOptim.batch_to_device(train_dataset, device)
            ),
            batch_size=params['batch_size'],
            collate_fn=ModuleOptim.dict_collate_fn,
            shuffle=True,
            num_workers=1,
        )

        kbest_scores = train_model(
            classifier, optimizer, kbest_scores,
            train_data_loader, val_data_loader, test_data_loader,
            vocab.get_token_to_index()['labels'],
            loss_func,
            acc_func,
            checkpoint_base,
            **params
        )

        logger.info("Kbest scores: %s" % kbest_scores)

    best_index = 0 if monitor.endswith('loss') else - 1

    best_checkpoint_file = ModuleOptim.get_checkpoint_filename(checkpoint_base, monitor, kbest_scores[best_index])

    best_checkpoint = torch.load(best_checkpoint_file,
                                 map_location=lambda storage,
                                 loc: storage)

    # logger.info("test_acc, mean: %.4f, stdev: %.4f" % (mean(test_acc_history), stdev(test_acc_history)))
    logger.info("Final best %s: %.4f" % (monitor,
                                         best_checkpoint['best_score']))
    logger.info("Final best params: %s" % best_checkpoint['params'])

    params = best_checkpoint['params']

    embedding_dict = prepare_embedding_dict(vocab, pretrained_weights, params)

    model, attn_layer, out_layer = prepare_model_components(link_type, vocab, params)

    classifier = TempModule2.TempClassifier(
        embedding_dict,
        model,
        attn_layer,
        out_layer
    ).to(device)

    classifier.load_state_dict(best_checkpoint['state_dict'])

    val_loss, val_acc, _, _ = TempEval.batch_eval_ET(
        classifier,
        val_data_loader,
        loss_func, acc_func, vocab.get_token_to_index()['labels'],
        param_space['update_label'][0],
        report=True
    )

    test_loss, test_acc, test_pred, test_targ = TempEval.batch_eval_ET(
        classifier,
        test_data_loader,
        loss_func, acc_func, vocab.get_token_to_index()['labels'],
        param_space['update_label'][0],
        report=True
    )

    print(len(pred_pkl), pred_pkl[0])

    TempUtils.save_to_pickle(test_pred, pred_pkl)
    TempUtils.save_to_pickle(test_targ, targ_pkl)


def train_model(classifier, optimizer, kbest_scores,
                train_data_loader, val_data_loader, test_data_loader,
                targ2ix,
                loss_func,
                acc_func,
                checkpoint_base, **params):

    monitor = params['monitor']

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

            train_targ = train_batch.pop('labels')
            train_feats = train_batch

            classifier.train()
            classifier.zero_grad()

            pred_prob = classifier(train_feats)

            train_loss = loss_func(pred_prob,
                                   train_targ,
                                   update_label=params['update_label'])

            train_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=params['max_norm'])
            optimizer.step()

            epoch_losses.append(train_loss.item())

            train_acc = acc_func(pred_prob,
                                 train_targ,
                                 update_label=params['update_label'])

            epoch_acces.append(train_acc)

            if (step != 0 and step % params['check_interval'] == 0) or step == step_num - 1:

                val_loss, val_acc, _, _ = TempEval.batch_eval_ET(
                    classifier,
                    val_data_loader,
                    loss_func, acc_func, targ2ix,
                    params['update_label']
                )

                test_loss, test_acc, _, _ = TempEval.batch_eval_ET(
                    classifier,
                    test_data_loader,
                    loss_func, acc_func,
                    targ2ix,
                    params['update_label']
                )

                eval_history['val_loss'].append(val_loss)
                eval_history['val_acc'].append(val_acc)

                monitor_score = round(locals()[monitor], 8)

                is_kbest, kbest_scores = ModuleOptim.update_kbest_scores(kbest_scores,
                                                                         monitor_score,
                                                                         monitor,
                                                                         kbest=params['kbest_checkpoint'])
                # print(kbest_scores)

                if is_kbest and len(kbest_scores) == params['kbest_checkpoint'] + 1:
                    removed_index = -1 if monitor.endswith('loss') else 0
                    removed_score = kbest_scores.pop(removed_index)
                    ModuleOptim.delete_checkpoint(ModuleOptim.get_checkpoint_filename(checkpoint_base,
                                                                                      monitor,
                                                                                      removed_score))
                    assert len(kbest_scores) == params['kbest_checkpoint']

                global_save_info = ModuleOptim.save_checkpoint({
                    'params': params,
                    'state_dict': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'monitor': monitor,
                    'best_score': monitor_score,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, is_kbest, ModuleOptim.get_checkpoint_filename(checkpoint_base,
                                                                 monitor,
                                                                 monitor_score))

                test_loss, test_acc, _, _ = TempEval.batch_eval_ET(
                    classifier,
                    test_data_loader,
                    loss_func, acc_func,
                    targ2ix,
                    params['update_label']
                )

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

    classification_model = 'Lstm2vec'   # 'baseRNN', 'attnRNN'

    update_label = 3

    param_space = {
        'classification_model': [classification_model],
        'update_label': [update_label],
        'freeze_mode': [False],
        'sdp_filter_nb': [100],     # SDP representation for each token
        'sdp_kernel_len': [3],
        'sdp_cnn_droprate': [0.3],
        'sdp_fc_dim': [100],
        'sdp_fc_droprate': [0.3],
        'dsdp_dim': [0],
        'dist_dim': [25],
        'input_dropout': [0.3],     # hyper-parameters of neural networks
        'rnn_hidden_dim': [200],
        'rnn_layer': [1],
        'rnn_dropout': [0.3],
        'use_attn':[True],
        'attn_fc_dim': [200],
        'attn_fc_drop': [0.3],
        'attn_out_drop': [0.3],
        'fc1_hidden_dim': [200],
        'fc1_dropout': [0.5],
        'batch_size': [32],
        'epoch_num': [1],
        'lr': [0.95e-0],           # hyper-parameters of optimizer
        'weight_decay': [1e-4],
        'max_norm': [4],
        'patience': [10],       # early stopping
        'monitor': ['val_acc'],
        'check_interval': [10],    # checkpoint based on val performance given a step interval
        'kbest_checkpoint': [5],
    }

    link_type = 'Event-DCT'

    data_dir = '20190222'

    train_datasets = ['TBD_TRAIN']

    val_datasets = ['TBD_VAL']

    test_datasets = ['TBD_TEST']

    dataset_flag = 'T:%s:V:%s:T:%s' % (
        '-'.join(train_datasets),
        '-'.join(val_datasets),
        '-'.join(test_datasets)
    )

    train_pkl = 'data/eventime/%s/%s/train_t_%s_l%i.pkl' % (
        data_dir,
        dataset_flag,
        link_type,
        update_label
    )

    val_pkl = 'data/eventime/%s/%s/val_t_%s_l%i.pkl' % (
        data_dir,
        dataset_flag,
        link_type,
        update_label
    )

    test_pkl = 'data/eventime/%s/%s/test_t_%s_l%i.pkl' % (
        data_dir,
        dataset_flag,
        link_type,
        update_label
    )

    info_pkl = 'data/eventime/%s/%s/glob_info_%s_l%i.pkl' % (
        data_dir,
        dataset_flag,
        link_type,
        update_label
    )

    embed_pkl = "data/eventime/%s/%s/giga.d200.%s.l%i.embed" % (
        data_dir,
        dataset_flag,
        link_type,
        update_label
    )

    pred_pkl = 'outputs/%s_%s_pred_%s_l%i.pkl' % (
        data_dir,
        dataset_flag,
        link_type,
        update_label
    )

    targ_pkl = 'outputs/%s_%s_targ_%s_l%i.pkl' % (
        data_dir,
        dataset_flag,
        link_type,
        update_label
    )

    optimize_model(link_type,
                   train_pkl, val_pkl, test_pkl,
                   info_pkl, embed_pkl, pred_pkl, targ_pkl, param_space,
                   max_evals=1)


if __name__ == '__main__':
    main()
