# coding=utf-8

import torch.utils.data as Data
import torch
from TempData import *
from TempModules import *
from sklearn.metrics import classification_report

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print('device:', device)

seed = 2
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def copyData2device(data, device):
    feat_dict, target = data
    feat_types = list(feat_dict.keys())
    feat_list = batch_to_device(list(feat_dict.values()), device)
    target = target.to(device=device)
    return dict(zip(feat_types, feat_list)), target


def preprocessData(**params):
    sent_win = params['sent_win']
    oper = params['oper_label']
    timeml_dir = os.path.join(os.path.dirname(__file__), "data/Timebank")
    anchor_file = os.path.join(os.path.dirname(__file__), "data/event-times_normalized.tab")
    pkl_file = os.path.join(os.path.dirname(__file__),
                            "data/unittest-%s-%s_w%i_%s.pkl" % (timeml_dir.split('/')[-1],
                                                                anchor_file.split('/')[-1],
                                                                sent_win,
                                                                'oper' if oper else 'order'))

    if params['doc_reset']:
        anchor_file2doc(timeml_dir, anchor_file, pkl_file, sent_win, oper=oper)

    doc_dic, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx, \
    max_sent_len, max_seq_len, max_mention_len, max_word_len = prepareGlobalSDP(pkl_file, link_types=['Event-DCT', 'Event-Timex'])

    return doc_dic, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx, \
           max_sent_len, max_seq_len, max_mention_len, max_word_len


def splitData(doc_dic, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
              max_sent_len, max_seq_len, max_mention_len, max_word_len, link_type):

    '''
    print train/dev/test feats shape
    '''
    train_data = feat2tensorSDP(doc_dic, TBD_TRAIN, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
                                max_sent_len, max_seq_len, max_mention_len, max_word_len, link_type)

    dev_data = feat2tensorSDP(doc_dic, TBD_DEV, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
                              max_sent_len, max_seq_len, max_mention_len, max_word_len, link_type)

    test_data = feat2tensorSDP(doc_dic, TBD_TEST, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
                               max_sent_len, max_seq_len, max_mention_len, max_word_len, link_type)
    return train_data, dev_data, test_data


def model_instance(wvocab_size, cvocab_size, pos_size, dep_size, dist_size, action_size,
                   max_sent_len, max_seq_len, max_mention_len, max_word_len,
                   pre_embed, verbose=0, **params):

    model = sentSdpRNN(wvocab_size, cvocab_size, pos_size, dep_size, dist_size, action_size,
                       max_sent_len, max_seq_len, max_mention_len, max_word_len,
                       pre_embed=pre_embed,
                       verbose=verbose, **params).to(device=device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params['lr'],
                           weight_decay=params['weight_decay'])

    return model, optimizer


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

        with torch.no_grad():
            dev_out = model(**dev_feat)
            dev_loss = F.nll_loss(dev_out, dev_target).item()
            dev_pred = torch.argmax(dev_out, dim=1)
            dev_acc = (dev_pred == dev_target).sum().item() / float(dev_pred.numel())
            print('epoch: %i, time: %.4f, '
                  'train loss: %.4f, train acc: %.4f | dev loss: %.4f, dev acc: %.4f' % (epoch,
                                                                                         time.time() - start_time,
                                                                                         sum(epoch_loss)/float(len(epoch_loss)),
                                                                                         sum(epoch_acc)/float(len(epoch_acc)),
                                                                                         dev_loss,
                                                                                         dev_acc))

    eval_data(model, test_feat, test_target, labels)


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

    link_type = 'Event-Timex'

    params = {
        'sent_win': 1,
        'oper_label': True,
        'link_type': link_type,
        'char_dim': 20,
        'pos_dim': 0,
        'dep_dim': 0,
        'dist_dim': 20,
        'seq_rnn_dim': 200,
        'sent_rnn_dim': 200,
        'dropout_sour_rnn': 0.5,
        'dropout_targ_rnn': 0.5,
        'dropout_sent_rnn': 0.5,
        'dropout_feat': 0.5,
        'mention_cat': 'sum',
        'fc_hidden_dim': 200,
        'seq_rnn_pool': True,
        'sent_rnn_pool': True,
        'sent_rnn': True,
        'sdp_rnn': True,
        'dropout_fc': 0.5,
        'batch_size': 32,
        'epoch_num': 20,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'max_norm': 5,
        'doc_reset': False,
        'data_reset': False,
    }
    pretrained_file = "Resources/embed/giga-aacw.d200.bin"
    pickle_embedding = "data/embedding.pkl"
    pickle_data = 'data/data_%s.pkl' % params['link_type']

    if params['data_reset']:
        doc_dic, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx, \
        max_sent_len, max_seq_len, max_mention_len, max_word_len = preprocessData(**params)

        word_idx, embedding = slimEmbedding(pretrained_file, pickle_embedding, word_idx, lowercase=False)

        train_data, dev_data, test_data = splitData(doc_dic, word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
                                                    max_sent_len, max_seq_len, max_mention_len, max_word_len, link_type)

        save_doc((train_data, dev_data, test_data,
                  word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx,
                  max_sent_len, max_seq_len, max_mention_len, max_word_len), pickle_data)

    train_data, dev_data, test_data, \
    word_idx, char_idx, pos_idx, dep_idx, dist_idx, rel_idx, \
    max_sent_len, max_seq_len, max_mention_len, max_word_len = load_doc(pickle_data)

    word_idx, embedding = load_doc(pickle_embedding)

    model, optimizer = model_instance(len(word_idx), len(char_idx), len(pos_idx), len(dep_idx), len(dist_idx), len(rel_idx),
                                      max_sent_len, max_seq_len, max_mention_len, max_word_len,
                                      pre_embed=embedding, verbose=0, **params)

    print(model)
    train_sdp_model(model, optimizer, train_data, dev_data, test_data, rel_idx, **params)


if __name__ == '__main__':
    main()