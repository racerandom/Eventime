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
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from TempModules import *
from TempData import *
from ModuleOptim import *
import TempUtils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print('device:', device)

seed = 2
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

is_pretrained = True
INPUT_COL, TARGET_COL = 0, 1


class TempOptimizer(nn.Module):

    def __init__(self, pkl_file, classifier, word_dim, epoch_nb, link_type, monitor, feat_types,
                 train_rate=1.0,
                 max_evals=500,
                 mode='tune',
                 pretrained_file='Resources/embed/giga-aacw.d200.bin',
                 screen_verbose=0):

        ## initialize the param space
        if classifier in ['CNN', 'AttnCNN']:
            self.param_space = {
                            'char_emb': [True],
                            'char_dim': range(20, 50 + 1, 10),
                            'kernel_len': [2, 3, 4, 5, 6, 7, 8],
                            'batch_size': [16, 32, 64, 128],
                            'pos_dim': range(10, 50 + 1, 10),
                            'dropout_emb': [0.0],
                            'filter_nb': range(100, 500 + 1, 20),
                            'dropout_conv': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                            'attn_targ': ['filter_nb', 'max_len'],
                            'mention_cat':['sum', 'max'],
                            'dropout_cat': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                            'dropout_fc': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                            'cat_word_tok': [True, False],
                            'cat_dist_tok': [True, False],
                            # 'optimizer': ['adadelta', 'adam','rmsprop', 'sgd'],
                            'fc_layer':[True, False],
                            'fc_hidden_dim': range(100, 500 + 1, 20),
                            'lr':[1e-2, 1e-3],
                            'weight_decay':[1e-3, 1e-4, 1e-5, 0]
                            }
        elif classifier in ['RNN', 'AttnRNN']:
            self.param_space = {
                'filter_nb': range(100, 500 + 1, 10),
                'char_in_dim': range(10, 50 + 1, 10),
                'char_hidden_dim': range(10, 50 + 1, 10),
                'batch_size': [16, 32, 64, 128],
                'fc_hidden_dim': range(100, 500 + 1, 10),
                'pos_dim': range(10, 50 + 1, 10),
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
        self.mode = mode
        self.feat_types = feat_types
        self.WORD_DIM = word_dim
        self.EPOCH_NUM = epoch_nb
        self.param_space['word_dim'] = [word_dim]

        self.doc_dic, self.word_idx, self.char_idx, self.pos_idx, self.rel_idx, \
        self.MAX_SEQ_LEN, self.MAX_TOK_LEN, self.MAX_CHAR_LEN, self.pre_model = prepare_global(pkl_file,
                                                                                               pretrained_file,
                                                                                               link_type=link_type)

        self.WVOCAB_SIZE = len(self.word_idx)
        self.CVOCAB_SIZE = len(self.char_idx)
        self.POS_SIZE = len(self.pos_idx)
        self.ACTION_SIZE = len(self.rel_idx)
        self.ACTIONS = [ key for key, value in sorted(self.rel_idx.items(), key=operator.itemgetter(1))]


        self.config = "lt=%s_m=%s_c=%s_pre=%s_r=%.2f_wd=%i_mo=%s_ep=%i_me=%i" % (
                                                                        link_type,
                                                                        self.mode,
                                                                        classifier,
                                                                        pretrained_file.split('/')[-1].split('.')[0] if pretrained_file else 'None',
                                                                        train_rate,
                                                                        word_dim,
                                                                        monitor,
                                                                        epoch_nb,
                                                                        max_evals)
        self.GLOB_BEST_MODEL_PATH = "models/glob_best_model_%s.pth" % self.config
        self.glob_best_score = None
        self.best_scores = []
        self.val_losses, self.val_acces = [], []
        self.test_losses, self.test_acces = [], []

        ## logger
        setup_logger(self.config, 'logs/%s.log' % self.config)
        self.logger = logging.getLogger(self.config)
        self.screen_verbose = screen_verbose


    def shape_dataset(self):

        self.logger.info('[Prepare train/dev/test data]')
        self.logger.info('train feat number: %i, target length: %i' % (len(self.train_feats), len(self.train_target)))
        self.logger.info('train feat shapes: %s' % [feat.shape for feat in self.train_feats])

        self.logger.info('dev feat number: %i, target length: %i' % (len(self.dev_feats), len(self.dev_target)))
        self.logger.info('dev feat shapes: %s' % [feat.shape for feat in self.dev_feats])

        self.logger.info('test feat number: %i, target length: %i' % (len(self.test_feats), len(self.test_target)))
        self.logger.info('test feat shapes: %s' % [feat.shape for feat in self.test_feats])


    def generate_feats_dataset(self, train_rate=1.0):


        ## split train/dev/test doc ids
        TRAIN_SET = sample_train(self.doc_dic.keys(), TA_DEV, TA_TEST, rate=train_rate)  # return training data
        DEV_SET = TA_DEV
        TEST_SET = TA_TEST
        self.logger.info("Train data: %i docs, Dev data: %i docs, Test data: %i docs..." % (len(TRAIN_SET),
                                                                                 len(DEV_SET),
                                                                                 len(TEST_SET)))

        ## prepare data based on doc ids
        self.train_feats, self.train_target = prepare_feats_dataset(self.doc_dic, TRAIN_SET, self.word_idx, self.char_idx, self.pos_idx, self.rel_idx,
                                                                    self.MAX_SEQ_LEN, self.MAX_TOK_LEN, self.MAX_CHAR_LEN,
                                                                    link_type=self.link_type, feat_types=self.feat_types)

        dev_feats, dev_target = prepare_feats_dataset(self.doc_dic, DEV_SET, self.word_idx, self.char_idx, self.pos_idx, self.rel_idx,
                                                      self.MAX_SEQ_LEN, self.MAX_TOK_LEN, self.MAX_CHAR_LEN,
                                                      link_type=self.link_type, feat_types=self.feat_types)

        test_feats, test_target = prepare_feats_dataset(self.doc_dic, TEST_SET, self.word_idx, self.char_idx, self.pos_idx, self.rel_idx,
                                                        self.MAX_SEQ_LEN, self.MAX_TOK_LEN, self.MAX_CHAR_LEN,
                                                        link_type=self.link_type, feat_types=self.feat_types)

        self.dev_feats = batch_to_device(dev_feats, device)
        self.dev_target = dev_target.to(device=device)

        self.test_feats = batch_to_device(test_feats, device)
        self.test_target = test_target.to(device=device)


    def optimize_model(self):

        for eval_i in range(1, self.max_evals + 1):
            params = {}
            for key, values in self.param_space.items():
                params[key] = random.choice(values)

            self.logger.info('[optim %i]: params: %s' % (eval_i, params))
            self.train_model(**params)

        # print(self.best_scores)
        # print(self.val_acces)
        # print(self.test_acces)

    def eval_best_model(self):

        checkpoint = (torch.load(self.GLOB_BEST_MODEL_PATH, map_location=lambda storage, loc: storage))

        model = TempClassifier(self.WVOCAB_SIZE, self.CVOCAB_SIZE, self.POS_SIZE, self.ACTION_SIZE,
                               self.MAX_SEQ_LEN, self.MAX_TOK_LEN, self.MAX_CHAR_LEN,
                               self.feat_types,
                               pre_model=self.pre_model,
                               verbose_level=self.screen_verbose,
                               **checkpoint['params']).to(device=device)

        model.load_state_dict(checkpoint['state_dict'])

        if self.screen_verbose:
            print(model)

        return self.eval_test(model, is_report=True, **checkpoint['params'])

    @staticmethod
    def eval_data(model, feats, target, action_labels, feat_types, is_report, logger=None, **params):

        model.eval()

        with torch.no_grad():
            out = model(feat_types, *feats, **params)
            loss = F.nll_loss(out, target).item()

            pred = torch.argmax(out, dim=1)
            acc = (pred == target).sum().item() / float(pred.numel())

            if is_report:
                logger.info('-' * 80)
                logger.info(classification_report(pred,
                                                  target,
                                                  target_names=action_labels))

            return loss, acc

    def eval_val(self, model, is_report=False, **params):

        return TempOptimizer.eval_data(model, self.dev_feats, self.dev_target, self.ACTIONS, self.feat_types, is_report, logger=self.logger, **params)

    def eval_test(self, model, is_report=False, **params):

        return TempOptimizer.eval_data(model, self.test_feats, self.test_target, self.ACTIONS, self.feat_types, is_report, logger=self.logger, **params)

    def eval_with_params(self, **params):

        train_dataset = MultipleDatasets(self.train_data[WORD_COL], self.train_data[POS_COL], self.train_data[REL_COL])

        train_data_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=1,
        )

        self.logger.info('Starting to train a new model with parameters', params)

        model = TempClassifier(self.WVOCAB_SIZE, self.CVOCAB_SIZE, self.POS_SIZE, self.ACTION_SIZE,
                               self.MAX_SEQ_LEN, self.MAX_TOK_LEN, self.MAX_CHAR_LEN,
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
        self.eval_val(model, **params)
        self.eval_test(model, **params)

    def train_model(self, **params):

        # print(self.feat_types)

        train_dataset = MultipleDatasets(*self.train_feats, self.train_target)

        train_data_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=1,
        )

        local_best_score, local_best_state = None, None

        model = TempClassifier(self.WVOCAB_SIZE, self.CVOCAB_SIZE, self.POS_SIZE, self.ACTION_SIZE,
                               self.MAX_SEQ_LEN, self.MAX_TOK_LEN, self.MAX_CHAR_LEN,
                               self.feat_types,
                               pre_model=self.pre_model,
                               verbose_level=self.screen_verbose,
                               **params).to(device=device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=params['lr'],
                               weight_decay=params['weight_decay'])  ##  fixed a error when using pre-trained embeddings

        if self.screen_verbose:
            print(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print('*', name)

        for epoch in range(1, self.EPOCH_NUM + 1):
            total_loss = []
            total_acc = []
            start_time = time.time()
            for step, train_sample in enumerate(train_data_loader):

                train_feats = batch_to_device(train_sample[:-1], device)
                train_target = train_sample[-1].to(device=device)

                model.train()
                # word_input, position_input, target = word_input.to(device), position_input.to(device), target.to(device)

                model.zero_grad()
                pred_out = model(self.feat_types, *train_feats, **params)
                loss = F.nll_loss(pred_out, train_target)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
                optimizer.step()
                total_loss.append(loss.data.item())
                pred = torch.argmax(pred_out, dim=1)
                total_acc.append((pred == train_target).sum().item() / float(pred.numel()))

            with torch.no_grad():
                dev_out = model(self.feat_types, *self.dev_feats, **params)
                dev_loss = F.nll_loss(dev_out, self.dev_target).item()
                dev_pred = torch.argmax(dev_out, dim=1)
                dev_acc = ( dev_pred == self.dev_target).sum().item() / float(dev_pred.numel())
                dev_score = dev_loss if self.monitor == 'val_loss' else dev_acc

                test_loss, test_acc = self.eval_test(model, **params)

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


            self.logger.info("Epoch %i ,loss: %.4f, accuracy: %.4f, %.5s seconds, dev loss: %.4f, dev accuracy: %.4f | test loss: %.4f, acc %.4f %s" % (
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

        # self.logger.info("Current params: %s" % params)
        self.logger.info("best loss of current params: %.4f, acc: %.4f | test loss: %.4f, acc %.4f" % (local_best_state['val_loss'],
                                                                                                   local_best_state['val_acc'],
                                                                                                   local_best_state['test_loss'],
                                                                                                   local_best_state['test_acc']))
        self.logger.info("params of glob best loss: %s, epoch: %s" % (glob_checkpoint['params'], glob_checkpoint['epoch']))
        self.logger.info("glob monitor %s, best_score: %.4f, loss: %.4f,  acc: %.4f', | test loss: %.4f, acc %.4f" % (
                                                                                        glob_checkpoint['monitor'],
                                                                                        glob_checkpoint['best_score'],
                                                                                        glob_checkpoint['val_loss'],
                                                                                        glob_checkpoint['val_acc'],
                                                                                        glob_checkpoint['test_loss'],
                                                                                        glob_checkpoint['test_acc']))
        self.logger.info("*" * 80)



def main():


    ## a pre-defined param set.
    params = {
              'char_emb': True,
              'char_dim': 50,
              'kernel_len': 4,
              'batch_size': 16,
              'fc_hidden_dim': 480,
              'attn_targ': 'filter_nb',
              'pos_dim': 50,
              'filter_nb': 480,
              'dropout_conv': 0.2,
              'dropout_emb': 0.0,
              'dropout_cat': 0.5,
              'dropout_fc': 0.5,
              'mention_cat': 'mean',
              'cat_word_tok': True,
              'cat_dist_tok': False,
              'fc_layer':False,
              'lr': 0.001,
              'weight_decay': 0.0001,
              'classifier': 'AttnCNN',
              'word_dim': 200}

    ## plot figure
    fig = plt.figure()
    fig.suptitle('train curve', fontsize=16)
    plt.xlabel('Training Data (percentage of all)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)


    classifier = "CNN"
    link_type = 'Event-Timex'
    pkl_file = "data/0531.pkl"
    word_dim = 200
    epoch_nb = 20
    mode = 'train'  # 'tune' or 'train'
    monitor = 'val_loss'
    feat_types = ['word_seq',
                  'char_seq',
                  'sour_dist_seq',
                  'targ_dist_seq',
                  'sour_word_tok',
                  'targ_word_tok',
                  'sour_dist_tok',
                  'targ_dist_tok'] if link_type in ['Event-Timex',
                                                    'Event-Event'] else ['word_seq',
                                                                         'char_seq',
                                                                         'sour_dist_seq',
                                                                         'sour_word_tok']


    piece_nb = 1

    train_per, train_acc = [], []
    for i in range(1, piece_nb + 1, 1):
        train_rate = 1 / piece_nb * i
        print("Link type:", link_type)
        print("[Traing data rate: %.2f]" % train_rate)
        train_per.append(train_rate)
        temp_extractor = TempOptimizer(pkl_file, classifier, word_dim, epoch_nb, link_type, monitor, feat_types,
                                       train_rate=train_rate,
                                       max_evals=200,
                                       mode = mode,
                                       # pretrained_file='Resources/embed/deps.words.bin',
                                       screen_verbose=2,
                                       )
        temp_extractor.generate_feats_dataset(train_rate=train_rate) ## prepare train, dev, test data for input to the model
        temp_extractor.shape_dataset()
        if mode in ['tune']:
            temp_extractor.optimize_model()
        elif mode in ['train']:
            temp_extractor.train_model(**params)
        best_loss, best_acc = temp_extractor.eval_best_model()
        train_acc.append(best_acc)
    plt.plot(train_per, train_acc, marker='^', label=classifier)

    # params = {'classifier':classifier, 'filter_nb': 120, 'kernel_len': 3, 'batch_size': 128, 'fc_hidden_dim': 240, 'pos_dim': 10, 'dropout_emb': 0.0, 'dropout_cat': 0.5, 'dropout_fc': 0.5, 'lr': 0.01, 'weight_decay': 1e-05, 'word_dim': 300}
    # temp_extractor.train_model(**params)
    # temp_extractor.eval_model()


    # plt.plot(train_percentage, train_acc, 'b^')
    fig.savefig('logs/%s.jpg' % temp_extractor.config)



if __name__ == '__main__':
    main()














