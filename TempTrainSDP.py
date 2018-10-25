# coding=utf-8
import warnings
warnings.simplefilter("ignore", UserWarning)

import torch.utils.data as Data
import torch
from TempData import *
from TempModules import *
from sklearn.metrics import classification_report
from statistics import mean, median, variance, stdev

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print('device:', device)

seed = 2
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def is_best_score(score, best_score, monitor):
    if not best_score:
        is_best = True
        best_score = score
    else:
        is_best = bool(score < best_score) if monitor == 'val_loss' else bool(score > best_score)
        best_score = score if is_best else best_score
    return is_best, best_score


def copyData2device(data, device):
    feat_dict, target = data
    feat_types = list(feat_dict.keys())
    feat_list = batch_to_device(list(feat_dict.values()), device)
    target = target.to(device=device)
    return dict(zip(feat_types, feat_list)), target


def preprocessData(task, sent_win, oper, doc_reset):
    timeml_dir = os.path.join(os.path.dirname(__file__), "data/Timebank")
    anchor_file = os.path.join(os.path.dirname(__file__), "data/event-times_normalized.tab")
    pkl_file = os.path.join(os.path.dirname(__file__),
                            "data/%s-%s-%s_w%i_%s.pkl" % (task,
                                                          timeml_dir.split('/')[-1],
                                                          anchor_file.split('/')[-1],
                                                          sent_win,
                                                          'oper' if oper else 'order'))

    if doc_reset:
        anchor_file2doc(timeml_dir, anchor_file, pkl_file, sent_win, oper=oper)

    doc_dic, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx, \
    max_sent_len, max_seq_len, max_mention_len, max_word_len = prepareGlobalSDP(pkl_file, task)

    return doc_dic, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx, \
           max_sent_len, max_seq_len, max_mention_len, max_word_len


def splitData(doc_dic, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
              max_sent_len, max_seq_len, max_mention_len, max_word_len, task):

    '''
    print train/dev/test feats shape
    '''
    train_data = feat2tensorSDP(doc_dic, TBD_TRAIN, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
                                max_sent_len, max_seq_len, max_mention_len, max_word_len, task)

    dev_data = feat2tensorSDP(doc_dic, TBD_DEV, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
                              max_sent_len, max_seq_len, max_mention_len, max_word_len, task)

    test_data = feat2tensorSDP(doc_dic, TBD_TEST, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
                               max_sent_len, max_seq_len, max_mention_len, max_word_len, task)
    return train_data, dev_data, test_data


def model_instance(wvocab_size, cvocab_size, pos_size, dep_size, dist_size, action_size,
                   max_sent_len, max_seq_len, max_mention_len, max_word_len,
                   pre_embed, verbose=1, **params):

    model = sentRNN(wvocab_size, cvocab_size, pos_size, dep_size, dist_size, action_size,
                       max_sent_len, max_seq_len, max_mention_len, max_word_len,
                       pre_embed=pre_embed,
                       verbose=verbose, **params).to(device=device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params['lr'],
                           weight_decay=params['weight_decay'])

    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('*', name)

    return model, optimizer


def optimize_model(pretrained_file, task, param_space, max_evals):

    embedding_corpus = pretrained_file.split('/')[-1].split('.')[0]
    pickle_embedding = "data/embedding_%s.pkl" % embedding_corpus
    pickle_data = 'data/task_%s-embedding_%s.pkl' % (param_space['link_type'], embedding_corpus)

    sent_win = param_space['sent_win'][0]
    oper = param_space['oper_label'][0]
    doc_reset = param_space['doc_reset'][0]
    data_reset = param_space['data_reset'][0]
    monitor = param_space['monitor'][0]

    if data_reset:
        doc_dic, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx, \
        max_sent_len, max_seq_len, max_mention_len, max_word_len = preprocessData(task, sent_win, oper, doc_reset)

        word_idx, embedding = slimEmbedding(pretrained_file, pickle_embedding, word_idx, lowercase=False)


        train_data, dev_data, test_data = splitData(doc_dic, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
                                                    max_sent_len, max_seq_len, max_mention_len, max_word_len, task)

        save_doc((train_data, dev_data, test_data,
                  word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
                  max_sent_len, max_seq_len, max_mention_len, max_word_len), pickle_data)

    word_idx, embedding = load_doc(pickle_embedding)

    train_data, dev_data, test_data, \
    word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx, \
    max_sent_len, max_seq_len, max_mention_len, max_word_len = load_doc(pickle_data)

    monitor_score, test_loss, test_acc, test_param = [], [], [], []

    for eval_i in range(1, max_evals + 1):
        params = {}
        for key, values in param_space.items():
            params[key] = random.choice(values)

        print('[Selected %i Params]:' % eval_i, params)

        model, optimizer = model_instance(sizeOfVocab(word_idx),
                                          sizeOfVocab(char_idx),
                                          sizeOfVocab(pos_idx),
                                          sizeOfVocab(dep_idx),
                                          sizeOfVocab(dist_idx),
                                          sizeOfVocab(rel_idx),
                                          max_sent_len,
                                          max_seq_len,
                                          max_mention_len,
                                          max_word_len,
                                          pre_embed=embedding,
                                          verbose=0,
                                          **params)

        local_monitor_score, local_test_loss, local_test_acc, local_param = train_sdp_model(model, optimizer, train_data, dev_data, test_data, rel_idx, **params)
        monitor_score.append(local_monitor_score)
        test_loss.append(local_test_loss)
        test_acc.append(local_test_acc)
        test_param.append(local_param)

        best_index = monitor_score.index(max(monitor_score) if monitor.endswith('acc') else min(monitor_score))
        print("Current best test_acc: %.4f" % test_acc[best_index])

    print("test_acc, mean: %.4f, stdev: %.4f" % (mean(test_acc), stdev(test_acc)))
    best_index = monitor_score.index(max(monitor_score) if monitor.endswith('acc') else min(monitor_score))
    print("Final best test_acc: %.4f" % test_acc[best_index])
    print("Final best params:", test_param[best_index])


def train_sdp_model(model, optimizer, train_data, dev_data, test_data, labels, **params):

    train_feat_dict, train_target = train_data
    train_feat_types = list(train_feat_dict.keys())
    train_feat_list = list(train_feat_dict.values())

    train_dataset = MultipleDatasets(*train_feat_list, train_target)

    train_data_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=1,
    )

    dev_feat, dev_target = copyData2device(dev_data, device)

    test_feat, test_target = copyData2device(test_data, device)

    local_best_score, local_best_state = None, None

    for epoch in range(1, params['epoch_num'] + 1):
        epoch_loss = []
        epoch_acc = []
        start_time = time.time()
        for step, train_sample in enumerate(train_data_loader):
            train_epoch_feats = batch_to_device(train_sample[:-1], device)
            train_epoch_target = train_sample[-1].to(device=device)
            train_feat_dict = dict(zip(train_feat_types, train_epoch_feats))

            model.train()
            model.zero_grad()

            pred_out = model(**train_feat_dict)
            loss = F.nll_loss(pred_out, train_epoch_target)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['max_norm'])
            optimizer.step()
            epoch_loss.append(loss.data.item())
            pred = torch.argmax(pred_out, dim=1)
            epoch_acc.append((pred == train_epoch_target).sum().item() / float(pred.numel()))

        model.eval()
        # with torch.no_grad():
        dev_out = model(**dev_feat)
        dev_loss = F.nll_loss(dev_out, dev_target).item()
        dev_pred = torch.argmax(dev_out, dim=1)
        dev_acc = (dev_pred == dev_target).sum().item() / float(dev_pred.numel())
        dev_score = dev_loss if params['monitor'] == 'val_loss' else dev_acc

        local_is_best, local_best_score = is_best_score(dev_score, local_best_score, params['monitor'])

        test_out = model(**test_feat)
        test_loss = F.nll_loss(test_out, test_target).item()
        test_pred = torch.argmax(test_out, dim=1)
        test_acc = (test_pred == test_target).sum().item() / float(test_pred.numel())

        print('epoch: %i, time: %.4f, '
              'train loss: %.4f, train acc: %.4f | '
              'dev loss: %.4f, dev acc: %.4f | '
              'test loss: %.4f, test acc: %.4f' % (epoch,
                                                   time.time() - start_time,
                                                   sum(epoch_loss)/float(len(epoch_loss)),
                                                   sum(epoch_acc)/float(len(epoch_acc)),
                                                   dev_loss,
                                                   dev_acc,
                                                   test_loss,
                                                   test_acc))
        if local_is_best:
            local_best_state = {'best_score': local_best_score,
                                'val_loss': dev_loss,
                                'val_acc': dev_acc,
                                'test_loss': test_loss,
                                'test_acc': test_acc,
                                'params': params}

        save_info = save_checkpoint({
            'epoch': epoch,
            'params': params,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'monitor': params['monitor'],
            'best_score': local_best_score,
            'val_loss': dev_loss,
            'val_acc': dev_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        }, local_is_best, "models/best.pth")

    local_checkpoint = torch.load("models/best.pth", map_location=lambda storage, loc: storage)
    print('Local best: val_loss %.4f, val_acc %.4f | test_loss %.4f, test_acc %.4f' % (local_checkpoint['val_loss'],
                                                                                       local_checkpoint['val_acc'],
                                                                                       local_checkpoint['test_loss'],
                                                                                       local_checkpoint['test_acc']))
    model.load_state_dict(local_checkpoint['state_dict'])

    eval_data(model, test_feat, test_target, labels)
    return local_best_score, local_checkpoint['test_loss'], local_checkpoint['test_acc'], local_checkpoint['params']


def eval_data(model, feat_dict, target, rel_idx):

    model.eval()

    with torch.no_grad():
        out = model(**feat_dict)
        loss = F.nll_loss(out, target).item()

        pred = torch.argmax(out, dim=1)
        acc = (pred == target).sum().item() / float(pred.numel())

        idx_set = list(set([ p.item() for p in pred]).union(set([ t.item() for t in target])))

        print('-' * 80)
        print(classification_report(pred,
                                    target,
                                    target_names=[key for key, value in rel_idx.items() if value in idx_set]
                                    )
              )
        print(loss, acc)


def main():

    task = 'day_len'

    param_space = {
        'sent_win': [1],
        'oper_label': [True],
        'link_type': [task],
        'init_weight': [None, 'xavier', 'kaiming'],
        'elmo': [False],
        'dropout_elmo': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        'char_dim': [0],
        'pos_dim': [0],
        'dep_dim': [0],
        'dist_dim': range(5, 30+1, 5),
        'seq_rnn_dim': range(100, 400+1, 10),
        'sent_rnn_dim': range(100, 400+1, 10),
        'sent_out_cat': ['max'],
        'sdp_out_cat': ['max'],
        'dropout_sour_rnn': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        'dropout_targ_rnn': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        'dropout_sent_rnn': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        'dropout_feat': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        'mention_cat': ['sum', 'max', 'mean'],
        'fc_hidden_dim': range(100, 400+1, 10),
        'sent_sdp': [True],
        'seq_rnn_pool': [True],
        'sent_rnn_pool': [False],
        'sent_rnn': [True],
        'sdp_rnn': [False],
        'lexical_feat': [False],
        'dropout_fc': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        'batch_size': [16, 32, 64, 128],
        'epoch_num': [20],
        'lr': [0.01, 0.001],
        'weight_decay': [0.0001, 0.00001],
        'max_norm': [1, 5, 10],
        'monitor': ['val_acc'],
        'doc_reset': [False],
        'data_reset': [False]
    }
    pretrained_file = "Resources/embed/GoogleNews-vectors-negative300.bin"

    optimize_model(pretrained_file, task, param_space, 10)


if __name__ == '__main__':
    main()