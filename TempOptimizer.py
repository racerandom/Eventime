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

    def __init__(self, classifier, word_dim, epoch_nb, rel_types, monitor, max_evals=500, pretrained_file='Resources/embed/giga-aacw.d200.bin'):

        ## model parameters
        self.monitor = monitor
        self.rel_types = rel_types
        self.max_evals=max_evals
        if classifier in ['CNN', 'AttnCNN']:
            self.param_space = {
                            'filter_nb': range(100, 500 + 1, 10),
                            'kernel_len': [2, 3, 4, 5, 6, 7, 8],
                            'batch_size': [32, 64, 128],
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
                'batch_size': [16],
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
        self.param_space['classifier'] = [classifier]
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
        self.GLOB_BEST_MODEL_PATH = "models/glob_best_model_c%s_wd%i_ep%i_me%i.pth" % (classifier,
                                                                            word_dim,
                                                                            epoch_nb,
                                                                            max_evals)
        self.glob_best_score = None
        self.best_scores = []
        self.val_losses = []
        self.val_acces = []
        self.test_losses = []
        self.test_acces = []


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


    def optimize_model(self):

        for eval_i in range(1, self.max_evals + 1):
            params = {}
            for key, values in self.param_space.items():
                params[key] = random.choice(values)

            print('[optim %i]:' % eval_i, ', params:', params)
            self.train_model(**params)

        print(self.best_scores)
        print(self.val_acces)
        print(self.test_acces)


    def eval_model(self):

        WORD_COL, POS_COL, REL_COL = 0, 1, 2

        checkpoint = (torch.load(self.GLOB_BEST_MODEL_PATH, map_location=lambda storage, loc: storage))

        model = TempClassifier(self.VOCAB_SIZE, self.POS_SIZE, self.ACTION_SIZE, self.MAX_LEN, pre_model=self.pre_model, **checkpoint['params']).to(device=device)
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=checkpoint['params']['lr'],
                               weight_decay=checkpoint['params']['weight_decay'])

        model.load_state_dict(checkpoint['state_dict'])

        print(self.eval_val(model))
        self.eval_test(model, is_report=True)


    def train_model(self, **params):

        WORD_COL, POS_COL, REL_COL = 0, 1, 2

        train_dataset = MultipleDatasets(self.train_data[WORD_COL], self.train_data[POS_COL], self.train_data[REL_COL])

        train_data_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=1,
        )

        # print('Starting to train a new model with parameters', params)

        local_best_score, local_best_state = None, None

        model = TempClassifier(self.VOCAB_SIZE, self.POS_SIZE, self.ACTION_SIZE, self.MAX_LEN, pre_model=self.pre_model,
                               **params).to(device=device)

        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params['lr'],
                               weight_decay=params['weight_decay'])  ##  fixed a error when using pre-trained embeddings

        print(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print('*', name)

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

                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         if name == 'temp_detector.fc2.weight':
                #             print(param[1][:5])

            model.eval()

            with torch.no_grad():
                dev_out = model(self.dev_data[WORD_COL], self.dev_data[POS_COL])
                dev_loss = F.nll_loss(dev_out, self.dev_data[REL_COL]).item()
                dev_diff = torch.eq(torch.argmax(dev_out, dim=1), self.dev_data[REL_COL])
                dev_acc = dev_diff.sum().item() / float(dev_diff.numel())
                dev_score = dev_loss if self.monitor == 'val_loss' else dev_acc

                test_loss, test_acc = self.eval_test(model)


            if not local_best_score:
                local_best_score = dev_score
                local_is_best = True
            else:
                local_is_best = bool(dev_score < local_best_score) if self.monitor == 'val_loss' else bool(dev_score > local_best_score)
                local_best_score = dev_score if local_is_best else local_best_score

            if local_is_best:
                local_best_state = {'best_score': local_best_score,
                                    'val_loss': dev_loss,
                                    'val_acc': dev_acc,
                                    'test_loss': test_loss,
                                    'test_acc':test_acc}


            if not self.glob_best_score:
                self.glob_best_score = dev_score
                glob_is_best = True
            else:
                glob_is_best = bool(dev_score < self.glob_best_score) if self.monitor == 'val_loss' else \
                    bool(dev_score > self.glob_best_score)
                self.glob_best_score = dev_score if glob_is_best else self.glob_best_score


            save_info = save_checkpoint({
                                        'epoch': epoch,
                                        'params': params,
                                        'state_dict': model.state_dict(),
                                        'monitor': self.monitor,
                                        'best_score': local_best_score,
                                        'val_loss': dev_loss,
                                        'val_acc': dev_acc,
                                        'test_loss': test_loss,
                                        'test_acc':test_acc
                                        }, glob_is_best, self.GLOB_BEST_MODEL_PATH)


            print('Epoch %i' % epoch, ',loss: %.4f' % (sum(total_loss) / float(len(total_loss))),
                  ', accuracy: %.4f' % (sum(total_acc) / float(len(total_acc))),
                  ', %.5s seconds' % (time.time() - start_time),
                  ', dev loss: %.4f' % (dev_loss),
                  ', dev accuracy: %.4f' % (dev_acc),
                  "| test loss: %.4f, acc %.4f" % (test_loss, test_acc),
                  save_info
                  )

        self.best_scores.append(local_best_state['best_score'])
        self.val_losses.append(local_best_state['val_loss'])
        self.val_acces.append(local_best_state['val_acc'])
        self.test_losses.append(local_best_state['test_loss'])
        self.test_acces.append(local_best_state['test_acc'])

        glob_checkpoint = torch.load(self.GLOB_BEST_MODEL_PATH,
                                map_location=lambda storage, loc: storage)

        print("Current params:", params)
        print("best loss of current params: %.4f" % local_best_state['val_loss'], ', acc: %.4f' % local_best_state['val_acc'],
              "| test loss: %.4f, acc %.4f" % (local_best_state['test_loss'], local_best_state['test_acc']))
        print("params of glob best loss:", glob_checkpoint['params'])
        print("glob monitor %s, best_score: %.4f, loss: %.4f,  acc: %.4f', | test loss: %.4f, acc %.4f" % (glob_checkpoint['monitor'],
                                                                                        glob_checkpoint['best_score'],
                                                                                        glob_checkpoint['val_loss'],
                                                                                        glob_checkpoint['val_acc'],
                                                                                        glob_checkpoint['test_loss'],
                                                                                        glob_checkpoint['test_acc']))
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

            return dev_loss, dev_acc


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

            return test_loss, test_acc




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

    classifier = "CNN"

    temp_extractor = TempOptimizer(classifier, 300, 5, ['Event-Timex', 'Timex-Event'], 'val_loss', max_evals=2,
                                   pretrained_file='Resources/embed/deps.words.bin')
    temp_extractor.optimize_model()
    temp_extractor.eval_model()
    # params = {'classifier':classifier, 'filter_nb': 120, 'kernel_len': 3, 'batch_size': 128, 'fc_hidden_dim': 240, 'pos_dim': 10, 'dropout_emb': 0.0, 'dropout_cat': 0.5, 'dropout_fc': 0.5, 'lr': 0.01, 'weight_decay': 1e-05, 'word_dim': 300}
    # temp_extractor.train_model(**params)
    # temp_extractor.eval_model()


if __name__ == '__main__':
    main()














