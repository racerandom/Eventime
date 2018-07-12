import warnings
warnings.simplefilter("ignore", UserWarning)
import logging

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from IPython.display import clear_output
import numpy as np
import random
import pdb
import gensim
import time, operator
from sklearn.metrics import classification_report

from TempModules import *
from TempData import *
import TempUtils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print('device:', device)

seed = 2
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

is_pretrained = True
INPUT_COL, TARGET_COL = 0, 1


def batch_to_device(inputs, device):
    for input in inputs:
        input.to(device=device)


def is_best_score(score, best_score, monitor):
    if not best_score:
        is_best = True
        best_score = score
    else:
        is_best = bool(score < best_score) if monitor == 'val_loss' else bool(score > best_score)
        best_score = score if is_best else best_score
    return is_best, best_score


class TempOptimizer(nn.Module):

    def __init__(self, pkl_file, classifier, word_dim, epoch_nb, link_type, monitor, feat_types, train_rate=1.0, max_evals=500, pretrained_file='Resources/embed/giga-aacw.d200.bin'):

        ## initialize the param space
        if classifier in ['CNN', 'AttnCNN']:
            self.param_space = {
                            'filter_nb': range(100, 500 + 1, 10),
                            'kernel_len': [2, 3, 4, 5, 6, 7, 8],
                            'batch_size': [16, 32, 64, 128],
                            'fc_hidden_dim': range(100, 500 + 1, 20),
                            'pos_dim': range(5, 30 + 1, 5),
                            'dropout_emb': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                            'dropout_cat': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                            'dropout_fc': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                            # 'optimizer': ['adadelta', 'adam','rmsprop', 'sgd'],
                            'lr':[1e-2, 1e-3],
                            'weight_decay':[1e-3, 1e-4, 1e-5, 0]
                            }
        elif classifier in ['RNN', 'AttnRNN']:
            self.param_space = {
                'filter_nb': range(100, 500 + 1, 10),
                'batch_size': [16, 32, 64, 128],
                'fc_hidden_dim': range(100, 500 + 1, 10),
                'pos_dim': range(5, 30 + 1, 5),
                'dropout_emb': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                'dropout_rnn': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                'dropout_cat': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                'dropout_fc': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                # 'optimizer': ['adadelta', 'adam','rmsprop', 'sgd'],
                'lr': [1e-1, 1e-2, 1e-3],
                'weight_decay': [1e-3, 1e-4, 1e-5, 0]
            }

        ## fixed parameters
        self.param_space['classifier'] = [classifier]
        self.monitor = monitor
        self.link_type = link_type
        self.max_evals = max_evals
        self.feat_types = feat_types
        self.WORD_DIM = word_dim
        self.EPOCH_NUM = epoch_nb
        self.param_space['word_dim'] = [word_dim]


        self.doc_dic, self.word_idx, self.pos_idx, self.rel_idx, self.max_len, self.max_token_len, self.pre_model = prepare_global(pkl_file,
                                                                                                               pretrained_file,
                                                                                                               link_type=link_type)
        self.VOCAB_SIZE = len(self.word_idx)
        self.POS_SIZE = len(self.pos_idx)
        self.MAX_LEN = self.max_len
        self.MAX_TOKEN_LEN = self.max_token_len
        self.ACTION_SIZE = len(self.rel_idx)
        self.ACTIONS = [ key for key, value in sorted(self.rel_idx.items(), key=operator.itemgetter(1))]


        ## Data and records
        self.TRAIN_SET = sample_train(self.doc_dic.keys(), TA_DEV, TA_TEST, rate=train_rate) # return training data
        self.DEV_SET = TA_DEV
        self.TEST_SET = TA_TEST
        print("Train data: %i docs, Dev data: %i docs, Test data: %i docs..." % (len(self.TRAIN_SET),
                                                                                 len(self.DEV_SET),
                                                                                 len(self.TEST_SET)))
        # self.train_data, self.dev_data, self.test_data = self.generate_data()

        self.config = "c=%s_pre=%s_r=%.1f_wd=%i_ep=%i_me=%i" % (classifier,
                                                                pretrained_file.split('/')[-1].split('.')[0],
                                                                train_rate,
                                                                word_dim,
                                                                epoch_nb,
                                                                max_evals)
        self.GLOB_BEST_MODEL_PATH = "models/glob_best_model_%s.pth" % self.config
        self.glob_best_score = None
        self.best_scores = []
        self.val_losses, self.val_acces = [], []
        self.test_losses, self.test_acces = [], []
        logging.basicConfig(filename='logs/%s.log' % self.config,
                            filemode='w',
                            level=logging.INFO)


    def generate_data(self):

        train_word_in, train_pos_in, train_rel_in = prepare_dataset(self.doc_dic, self.TRAIN_SET, self.word_idx, self.pos_idx, self.rel_idx,
                                                                 self.MAX_LEN, link_type=self.link_type)

        dev_word_in, dev_pos_in, dev_rel_in = prepare_dataset(self.doc_dic, self.DEV_SET, self.word_idx, self.pos_idx, self.rel_idx, self.MAX_LEN,
                                                           link_type=self.link_type)

        test_word_in, test_pos_in, test_rel_in = prepare_dataset(self.doc_dic, self.TEST_SET, self.word_idx, self.pos_idx, self.rel_idx, self.MAX_LEN,
                                                              link_type=self.link_type)

        train_data = (
            train_word_in,
            train_pos_in,
            train_rel_in
        )

        dev_data = (dev_word_in.to(device=device),
                    dev_pos_in.to(device=device),
                    dev_rel_in.to(device=device)
        )

        test_data = (test_word_in.to(device=device),
                     test_pos_in.to(device=device),
                     test_rel_in.to(device=device)
        )

        return train_data, dev_data, test_data


    def shape_dataset(self):

        print('train feat number: %i, target length: %i' % (len(self.train_feats), len(self.train_target)))
        print('train feat shapes:', [feat.shape for feat in self.train_feats])

        print('dev feat number: %i, target length: %i' % (len(self.dev_feats), len(self.dev_target)))
        print('dev feat shapes:', [feat.shape for feat in self.dev_feats])

        print('test feat number: %i, target length: %i' % (len(self.test_feats), len(self.test_target)))
        print('test feat shapes:', [feat.shape for feat in self.test_feats])


    def generate_feats_dataset(self):

        self.train_feats, self.train_target = prepare_feats_dataset(self.doc_dic, self.TRAIN_SET, self.word_idx, self.pos_idx, self.rel_idx,
                                                self.MAX_LEN, self.max_token_len, link_type=self.link_type, feat_types=self.feat_types)

        self.dev_feats, self.dev_target = prepare_feats_dataset(self.doc_dic, self.DEV_SET, self.word_idx, self.pos_idx, self.rel_idx, self.MAX_LEN,
                                              self.max_token_len, link_type=self.link_type, feat_types=self.feat_types)

        self.test_feats, self.test_target = prepare_feats_dataset(self.doc_dic, self.TEST_SET, self.word_idx, self.pos_idx, self.rel_idx, self.MAX_LEN,
                                               self.max_token_len, link_type=self.link_type, feat_types=self.feat_types)

        batch_to_device(self.dev_feats, device)
        self.dev_target.to(device=device)
        print('dev target is cuda:', self.dev_target.is_cuda)

        batch_to_device(self.test_feats, device)
        self.test_target.to(device=device)
        print('test target is cuda:', self.test_target.is_cuda)




    def optimize_model(self):

        for eval_i in range(1, self.max_evals + 1):
            params = {}
            for key, values in self.param_space.items():
                params[key] = random.choice(values)

            logging.info('[optim %i]: params: %s' % (eval_i, params))
            self.train_model(**params)

        # print(self.best_scores)
        # print(self.val_acces)
        # print(self.test_acces)


    def eval_best_model(self):

        checkpoint = (torch.load(self.GLOB_BEST_MODEL_PATH, map_location=lambda storage, loc: storage))

        model = TempClassifier(self.VOCAB_SIZE, self.POS_SIZE, self.ACTION_SIZE, self.MAX_LEN, pre_model=self.pre_model, **checkpoint['params']).to(device=device)

        model.load_state_dict(checkpoint['state_dict'])

        # print(self.eval_val(model))
        self.eval_test(model, is_report=True)

    @staticmethod
    def eval_data(model, feats, target, action_labels, feat_types, is_report):

        model.eval()

        with torch.no_grad():
            out = model(feat_types, *feats)
            loss = F.nll_loss(out, target).item()

            pred = torch.argmax(out, dim=1)
            acc = (pred == target).sum().item() / float(pred.numel())

            if is_report:
                logging.info('-' * 80)
                logging.info(classification_report(pred, target,
                                            target_names=action_labels))

            return loss, acc

    def eval_val(self, model, is_report=False):

        return TempOptimizer.eval_data(model, self.dev_feats, self.dev_target, self.ACTIONS, self.feat_types, is_report)

    def eval_test(self, model, is_report=False):

        return TempOptimizer.eval_data(model, self.test_feats, self.test_target, self.ACTIONS, self.feat_types, is_report)

    def eval_with_params(self, **params):

        train_dataset = MultipleDatasets(self.train_data[WORD_COL], self.train_data[POS_COL], self.train_data[REL_COL])

        train_data_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=1,
        )

        print('Starting to train a new model with parameters', params)

        model = TempClassifier(self.VOCAB_SIZE,
                               self.POS_SIZE,
                               self.ACTION_SIZE,
                               self.MAX_LEN,
                               self.feat_types,
                               pre_model=self.pre_model,
                               **params).to(device=device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params['lr'],
                               weight_decay=params['weight_decay'])

        for epoch in range(1, params['best_epoch'] + 1):
            for step, (word_input, position_input, target) in enumerate(train_data_loader):

                ## switch to train mode
                model.train()
                word_input, position_input, target = word_input.to(device), position_input.to(device), target.to(device)

                model.zero_grad()
                pred_out = model(word_input, position_input)
                loss = F.nll_loss(pred_out, target)
                loss.backward(retain_graph=True)
                optimizer.step()


        ## switch to eval mode
        self.eval_val(model)
        self.eval_test(model)

    def train_model(self, **params):

        print(self.feat_types)

        train_dataset = MultipleDatasets(*self.train_feats, self.train_target)

        train_data_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=1,
        )

        local_best_score, local_best_state = None, None

        model = TempClassifier(self.VOCAB_SIZE, self.POS_SIZE, self.ACTION_SIZE, self.MAX_LEN, self.MAX_TOKEN_LEN, self.feat_types, pre_model=self.pre_model,
                               **params).to(device=device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=params['lr'],
                               weight_decay=params['weight_decay'])  ##  fixed a error when using pre-trained embeddings

        # print(model)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print('*', name)

        for epoch in range(1, self.EPOCH_NUM + 1):
            total_loss = []
            total_acc = []
            start_time = time.time()
            for step, train_sample in enumerate(train_data_loader):

                train_feats, train_target = train_sample[:-1], train_sample[-1]
                batch_to_device(train_feats, device)
                train_target.to(device=device)

                model.train()
                # word_input, position_input, target = word_input.to(device), position_input.to(device), target.to(device)

                model.zero_grad()
                pred_out = model(self.feat_types, *train_feats)
                loss = F.nll_loss(pred_out, train_target)
                loss.backward(retain_graph=True)
                optimizer.step()
                total_loss.append(loss.data.item())
                pred = torch.argmax(pred_out, dim=1)
                total_acc.append((pred == train_target).sum().item() / float(pred.numel()))

            model.eval()

            with torch.no_grad():
                dev_out = model(self.feat_types, *self.dev_feats)
                dev_loss = F.nll_loss(dev_out, self.dev_target).item()
                dev_pred = torch.argmax(dev_out, dim=1)
                dev_acc = ( dev_pred == self.dev_target).sum().item() / float(dev_pred.numel())
                dev_score = dev_loss if self.monitor == 'val_loss' else dev_acc

                test_loss, test_acc = self.eval_test(model)

            local_is_best, local_best_score = is_best_score(dev_score, local_best_score, self.monitor)

            if local_is_best:
                local_best_state = {'best_score': local_best_score,
                                    'val_loss': dev_loss,
                                    'val_acc': dev_acc,
                                    'test_loss': test_loss,
                                    'test_acc':test_acc,
                                    'params': params}

            glob_is_best, self.glob_best_score = is_best_score(dev_score, self.glob_best_score, self.monitor)

            save_info = save_checkpoint({
                                        'epoch': epoch,
                                        'params': params,
                                        'state_dict': model.state_dict(),
                                        'optimizer': optimizer.state_dict(),
                                        'monitor': self.monitor,
                                        'best_score': local_best_score,
                                        'val_loss': dev_loss,
                                        'val_acc': dev_acc,
                                        'test_loss': test_loss,
                                        'test_acc':test_acc
                                        }, glob_is_best, self.GLOB_BEST_MODEL_PATH)


            logging.info("Epoch %i ,loss: %.4f, accuracy: %.4f, %.5s seconds, dev loss: %.4f, dev accuracy: %.4f | test loss: %.4f, acc %.4f %s" % (
                                                                   epoch,
                                                                   sum(total_loss) / float(len(total_loss)),
                                                                   sum(total_acc) / float(len(total_acc)),
                                                                   time.time() - start_time,
                                                                   dev_loss,
                                                                   dev_acc,
                                                                   test_loss,
                                                                   test_acc,
                                                                   save_info))

        self.best_scores.append(local_best_state['best_score'])
        self.val_losses.append(local_best_state['val_loss'])
        self.val_acces.append(local_best_state['val_acc'])
        self.test_losses.append(local_best_state['test_loss'])
        self.test_acces.append(local_best_state['test_acc'])

        glob_checkpoint = torch.load(self.GLOB_BEST_MODEL_PATH,
                                map_location=lambda storage, loc: storage)

        logging.info("Current params: %s" % params)
        logging.info("best loss of current params: %.4f, acc: %.4f | test loss: %.4f, acc %.4f" % (local_best_state['val_loss'],
                                                                                                   local_best_state['val_acc'],
                                                                                                   local_best_state['test_loss'],
                                                                                                   local_best_state['test_acc']))
        logging.info("params of glob best loss: %s" % glob_checkpoint['params'])
        logging.info("glob monitor %s, best_score: %.4f, loss: %.4f,  acc: %.4f', | test loss: %.4f, acc %.4f" % (
                                                                                        glob_checkpoint['monitor'],
                                                                                        glob_checkpoint['best_score'],
                                                                                        glob_checkpoint['val_loss'],
                                                                                        glob_checkpoint['val_acc'],
                                                                                        glob_checkpoint['test_loss'],
                                                                                        glob_checkpoint['test_acc']))
        logging.info("*" * 80)



def main():

    classifier = "CNN"
    link_type = 'Event-Timex'
    pkl_file = "data/0531_%s.pkl" % (link_type)
    word_dim = 300
    epoch_nb = 1
    monitor = 'val_loss'
    feat_types = ['token_seq',
                  'sour_dist_seq',
                  'targ_dist_seq',
                  'sour_token',
                  'targ_token',
                  'sour_dist',
                  'targ_dist']

    temp_extractor = TempOptimizer(pkl_file, classifier, word_dim, epoch_nb, link_type, monitor, feat_types,
                                   train_rate=1.0,
                                   max_evals=1,
                                   pretrained_file='Resources/embed/deps.words.bin'
                                   )
    temp_extractor.generate_feats_dataset() ## prepare train, dev, test data for input to the model
    temp_extractor.shape_dataset()
    temp_extractor.optimize_model()
    # temp_extractor.eval_best_model()
    # params = {'classifier':classifier, 'filter_nb': 120, 'kernel_len': 3, 'batch_size': 128, 'fc_hidden_dim': 240, 'pos_dim': 10, 'dropout_emb': 0.0, 'dropout_cat': 0.5, 'dropout_fc': 0.5, 'lr': 0.01, 'weight_decay': 1e-05, 'word_dim': 300}
    # temp_extractor.train_model(**params)
    # temp_extractor.eval_model()


if __name__ == '__main__':
    main()














