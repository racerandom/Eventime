import warnings
warnings.simplefilter("ignore", UserWarning)

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

class TempOptimizer(nn.Module):

    def __init__(self, word_dim, epoch_nb, rel_types, monitor, pretrained_file='Resources/embed/giga-aacw.d200.bin'):

        ## model parameters
        self.monitor = monitor
        self.rel_types = rel_types
        self.param_space = {
                            'classifier':[
                                          'RNN',
                                          # 'CNN',
                                          # 'AttnCNN',
                                          # 'AttnRNN',
                                          ],
                            'filter_nb': range(100, 500 + 1, 10),
                            'kernel_len': [2, 3, 4, 5],
                            'batch_size': [32, 64, 128],
                            'fc_hidden_dim': range(100, 500 + 1, 10),
                            'pos_dim': range(5, 30 + 1, 5),
                            'dropout_emb': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                            'dropout_rnn': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                            'dropout_cat': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                            'dropout_fc': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                            # 'optimizer': ['adadelta', 'adam','rmsprop', 'sgd'],
                            'lr':[1e-2, 1e-3],
                            'weight_decay':[1e-3, 1e-4, 1e-5, 0]
                            }
        self.doc_dic, self.word_idx, self.pos_idx, self.rel_idx, self.max_len, self.pre_model = prepare_global(pretrained_file, types=rel_types)
        self.VOCAB_SIZE = len(self.word_idx)
        self.POS_SIZE = len(self.pos_idx)
        self.MAX_LEN = self.max_len
        self.ACTION_SIZE = len(self.rel_idx)
        self.ACTIONS = [ key for key, value in sorted(self.rel_idx.items(), key=operator.itemgetter(1))]
        self.WORD_DIM = word_dim
        self.EPOCH_NUM = epoch_nb
        self.param_space['word_dim'] = [word_dim]

        ## Data and records
        self.train_data, self.dev_data, self.test_data = self.generate_data()
        self.GLOB_BEST_MODEL_PATH = "models/glob_best_model.pth"
        self.glob_best_acc = 0
        self.glob_best_loss = 1000
        self.glob_best_params = {}
        self.glob_test = ""


    def generate_data(self):

        train_word_in, train_pos_in, train_rel_in = prepare_data(self.doc_dic, TBD_TRAIN, self.word_idx, self.pos_idx, self.rel_idx,
                                                                 self.MAX_LEN, types=self.rel_types)

        dev_word_in, dev_pos_in, dev_rel_in = prepare_data(self.doc_dic, TBD_DEV, self.word_idx, self.pos_idx, self.rel_idx, self.MAX_LEN,
                                                           types=self.rel_types)

        test_word_in, test_pos_in, test_rel_in = prepare_data(self.doc_dic, TBD_TEST, self.word_idx, self.pos_idx, self.rel_idx, self.MAX_LEN,
                                                              types=self.rel_types)

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


    def optimize_model(self, max_evals=200):

        for eval_i in range(1, max_evals + 1):
            params = {}
            for key, values in self.param_space.items():
                params[key] = random.choice(values)

            print('[optim %i]:' % eval_i, ', params:', params)
            self.train_model(**params)


    def eval_model(self):

        WORD_COL, POS_COL, REL_COL = 0, 1, 2

        model = TempClassifier(self.VOCAB_SIZE, self.POS_SIZE, self.ACTION_SIZE, self.MAX_LEN, pre_model=self.pre_model, **self.glob_best_params).to(device=device)
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.glob_best_params['lr'],
                               weight_decay=self.glob_best_params['weight_decay'])

        model.load_state_dict(torch.load(self.GLOB_BEST_MODEL_PATH))

        print(self.eval_val(model))
        print(self.eval_test(model, is_report=True))


    def train_model(self, **params):

        WORD_COL, POS_COL, REL_COL = 0, 1, 2

        train_dataset = MultipleDatasets(self.train_data[WORD_COL], self.train_data[POS_COL], self.train_data[REL_COL])

        train_data_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=1,
        )

        print('Starting to train a new model with parameters', params)

        local_best_acc, local_best_loss, local_best_epoch, local_best_test = 0, 1000, 0, ""
        # BEST_MODEL_PATH = "models/best-model_%s.pth" % (TempUtils.dict2str(params))
        BEST_MODEL_PATH = "models/temp-model.pth"

        model = TempClassifier(self.VOCAB_SIZE, self.POS_SIZE, self.ACTION_SIZE, self.MAX_LEN, pre_model=self.pre_model,
                               **params).to(device=device)

        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params['lr'],
                               weight_decay=params['weight_decay'])  ##  fixed a error when using pre-trained embeddings

        # print(model)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print('*', name)

        for epoch in range(1, self.EPOCH_NUM + 1):
            total_loss = []
            total_acc = []
            start_time = time.time()
            for step, (word_input, position_input, target) in enumerate(train_data_loader):
                model.train()
                word_input, position_input, target = word_input.to(device), position_input.to(device), target.to(device)

                model.zero_grad()
                pred_out = model(word_input, position_input)
                loss = loss_function(pred_out, target)
                loss.backward(retain_graph=True)
                optimizer.step()
                total_loss.append(loss.data.item())
                diff = torch.eq(torch.argmax(pred_out, dim=1), target)
                total_acc.append(diff.sum().item() / float(diff.numel()))

            model.eval()

            with torch.no_grad():
                dev_out = model(self.dev_data[WORD_COL], self.dev_data[POS_COL])
                dev_loss = F.nll_loss(dev_out, self.dev_data[REL_COL]).item()
                dev_diff = torch.eq(torch.argmax(dev_out, dim=1), self.dev_data[REL_COL])
                dev_acc = dev_diff.sum().item() / float(dev_diff.numel())

            if self.monitor == 'val_acc':
                if dev_acc > local_best_acc:
                    local_best_acc = dev_acc
                    local_best_loss = dev_loss
                    local_best_epoch = epoch
                    local_best_test = self.eval_test(model)
                    torch.save(model.state_dict(), BEST_MODEL_PATH)

            elif self.monitor == 'val_loss':
                if dev_loss < local_best_loss:
                    local_best_loss = dev_loss
                    local_best_acc = dev_acc
                    local_best_epoch = epoch
                    local_best_test = self.eval_test(model)
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
            else:
                raise Exception('Wrong monitor parameter...')

            print('Epoch %i' % epoch, ',loss: %.4f' % (sum(total_loss) / float(len(total_loss))),
                  ', accuracy: %.4f' % (sum(total_acc) / float(len(total_acc))),
                  ', %.5s seconds' % (time.time() - start_time),
                  ', dev loss: %.4f' % (dev_loss),
                  ', dev accuracy: %.4f' % (dev_acc),
                  self.eval_test(model)
                  )

        if self.monitor == 'val_acc':
            if self.glob_best_acc < local_best_acc:
                self.glob_best_acc = local_best_acc
                self.glob_best_loss = local_best_loss
                self.glob_test = local_best_test
                params['best_epoch'] = local_best_epoch
                self.glob_best_params = params
                model.load_state_dict(torch.load(BEST_MODEL_PATH))
                torch.save(model.state_dict(), self.GLOB_BEST_MODEL_PATH)

        elif self.monitor == 'val_loss':
            if self.glob_best_loss > local_best_loss:
                self.glob_best_loss = local_best_loss
                self.glob_best_acc = local_best_acc
                self.glob_test = local_best_test
                params['best_epoch'] = local_best_epoch
                self.glob_best_params = params
                model.load_state_dict(torch.load(BEST_MODEL_PATH))
                torch.save(model.state_dict(), self.GLOB_BEST_MODEL_PATH)
        else:
            raise Exception('Wrong monitor parameter...')

        print("Current params:", params)
        print("best loss of current params: %.4f" % (local_best_loss), ', acc: %.4f' % (local_best_acc), local_best_test)
        print("params of glob best loss:", self.glob_best_params)
        print("glob best loss: %.4f" % (self.glob_best_loss), ', acc: %.4f' % (self.glob_best_acc), self.glob_test)
        print("*" * 80)

    def eval_val(self, model, is_report=False):

        WORD_COL, POS_COL, REL_COL = 0, 1, 2

        model.eval()

        with torch.no_grad():
            dev_out = model(self.dev_data[WORD_COL], self.dev_data[POS_COL])
            dev_loss = F.nll_loss(dev_out, self.dev_data[REL_COL]).item()
            dev_diff = torch.eq(torch.argmax(dev_out, dim=1), self.dev_data[REL_COL])
            dev_acc = dev_diff.sum().item() / float(dev_diff.numel())

            if is_report:
                print(classification_report(torch.argmax(dev_out, dim=1), self.dev_data[REL_COL],
                                            target_names=self.ACTIONS))

            return " | dev loss: %.4f, dev acc: %.4f" % (dev_loss, dev_acc)


    def eval_test(self, model, is_report=False):

        WORD_COL, POS_COL, REL_COL = 0, 1, 2

        model.eval()

        with torch.no_grad():
            test_out = model(self.test_data[WORD_COL], self.test_data[POS_COL])
            test_loss = F.nll_loss(test_out, self.test_data[REL_COL]).item()
            test_diff = torch.eq(torch.argmax(test_out, dim=1), self.test_data[REL_COL])
            test_acc = test_diff.sum().item() / float(test_diff.numel())

            if is_report:
                print('-' * 80)
                print(classification_report(torch.argmax(test_out, dim=1), self.test_data[REL_COL],
                                            target_names=self.ACTIONS))

            return " | test loss: %.4f, test acc: %.4f" % (test_loss, test_acc)




    def eval_with_params(self, **params):

        WORD_COL, POS_COL, REL_COL = 0, 1, 2

        train_dataset = MultipleDatasets(self.train_data[WORD_COL], self.train_data[POS_COL], self.train_data[REL_COL])

        train_data_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=1,
        )

        print('Starting to train a new model with parameters', params)

        model = TempClassifier(self.VOCAB_SIZE, self.POS_SIZE, self.ACTION_SIZE, self.MAX_LEN, pre_model=self.pre_model,
                               **params).to(device=device)
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params['lr'],
                               weight_decay=params['weight_decay'])

        for epoch in range(1, params['best_epoch'] + 1):
            for step, (word_input, position_input, target) in enumerate(train_data_loader):

                ## switch to train mode
                model.train()
                word_input, position_input, target = word_input.to(device), position_input.to(device), target.to(device)

                model.zero_grad()
                pred_out = model(word_input, position_input)
                loss = loss_function(pred_out, target)
                loss.backward(retain_graph=True)
                optimizer.step()


        ## switch to eval mode
        self.eval_val(model)
        self.eval_test(model)

def main():

    temp_extractor = TempOptimizer(300, 10, ['Event-Timex', 'Timex-Event'], 'val_loss',
                                   pretrained_file='Resources/embed/deps.words.bin'
                                   )
    temp_extractor.optimize_model(max_evals=1)
    temp_extractor.eval_model()
    # params = {'filter_nb': 120, 'kernel_len': 3, 'batch_size': 128, 'fc_hidden_dim': 370, 'pos_dim': 5, 'dropout_emb': 0.45, 'dropout_cat': 0.55, 'lr': 0.001, 'weight_decay': 1e-05, 'word_dim': 200, 'best_epoch': 19}
    # temp_extractor.eval_with_params(**params)


if __name__ == '__main__':
    main()














