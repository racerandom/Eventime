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
import time
from sklearn.metrics import classification_report

from TempModules import *
from TempData import *
import TempUtils

torch.manual_seed(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

is_pretrained = True

class TempOptimizer(nn.Module):

    def __init__(self, rel_types, monitor):


        ## model parameters
        self.monitor = monitor
        self.rel_types = rel_types
        self.param_space = {
            'filter_nb': range(100, 500 + 1, 10),
            'kernel_len': [2, 3, 4, 5],
            'batch_size': [32, 64, 128],
            'fc_hidden_dim': range(100, 500 + 1, 10),
            'pos_dim': range(5, 30 + 1),
            'dropout_emb': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
            'dropout_max': [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
            'optimizer': ['adadelta', 'adam','rmsprop', 'sgd'],
            'lr':[1e-1, 1e-2, 1e-3],
            'weight_decay':[1e-2, 1e-3, 1e-4, 1e-5, 0]
            }
        self.doc_dic, self.word_idx, self.pos_idx, self.rel_idx, self.max_len, self.pre_model = prepare_global(is_pretrained=True, types=rel_types)
        self.VOCAB_SIZE = len(self.word_idx)
        self.POS_SIZE = len(self.pos_idx)
        self.MAX_LEN = self.max_len
        self.ACTION_SIZE = len(self.rel_idx)
        self.WORD_DIM = 200
        self.EPOCH_NUM = 20

        ## Data and records
        self.train_data, self.dev_data, self.test_data = self.generate_data()
        self.glob_best_acc = 0
        self.glob_best_loss = 100
        self.glob_best_params = {}


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



    def optimize_model(self, max_evals=100):

        for eval_i in range(max_evals):
            params = {}
            for key, values in self.param_space.items():
                params[key] = random.choice(values)

            print('eval %i:' % eval_i, ', params:', params)
            self.train_model(**params)


    def train_model(self, **args):

        WORD_COL, POS_COL, REL_COL = 0, 1, 2

        train_dataset = MultipleDatasets(self.train_data[WORD_COL], self.train_data[POS_COL], self.train_data[REL_COL])

        train_data_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=1,
        )


        print('Starting to train a new model with parameters', args)

        best_acc, best_loss = 0, 100
        GLOB_BEST_MODEL_PATH = "models/glob_best_model.pth"
        BEST_MODEL_PATH = "models/best-model_%s.pth" % (TempUtils.dict2str(args))

        model = TempClassifier(self.WORD_DIM, args['pos_dim'], args['filter_nb'], self.VOCAB_SIZE, self.POS_SIZE,
                               self.MAX_LEN, args['fc_hidden_dim'], self.ACTION_SIZE,
                               args['batch_size'], args['kernel_len'], pre_model=self.pre_model).to(device=device)

        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['lr'],
                               weight_decay=args['weight_decay'])  ##  fixed a error when using pre-trained embeddings
        # print(model)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print('*', name)

        for epoch in range(self.EPOCH_NUM):
            total_loss = []
            total_acc = []
            start_time = time.time()
            for step, (word_input, position_input, target) in enumerate(train_data_loader):
                word_input, position_input, target = word_input.to(device), position_input.to(device), target.to(device)

                model.zero_grad()
                pred_out = model(word_input, position_input)
                loss = loss_function(pred_out, target)
                loss.backward(retain_graph=True)
                optimizer.step()
                total_loss.append(loss.data.item())
                diff = torch.eq(torch.argmax(pred_out, dim=1), target)
                total_acc.append(diff.sum().item() / float(diff.numel()))

            dev_out = model(self.dev_data[WORD_COL], self.dev_data[POS_COL])
            dev_loss = F.nll_loss(dev_out, self.dev_data[REL_COL])
            dev_diff = torch.eq(torch.argmax(dev_out, dim=1), self.dev_data[REL_COL])
            dev_acc = dev_diff.sum().item() / float(dev_diff.numel())

            if self.monitor == 'val_acc':
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    best_loss = dev_loss
                    torch.save(model, BEST_MODEL_PATH)
            elif self.monitor == 'val_loss':
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    best_acc = dev_acc
                    torch.save(model, BEST_MODEL_PATH)
            else:
                raise Exception('Wrong monitor parameter...')
            #    print(classification_report(torch.argmax(dev_out, dim=1), dev_rel_in, labels=np.unique(torch.argmax(dev_out, dim=1))))

            print('Epoch %i' % epoch, ',loss: %.4f' % (sum(total_loss) / float(len(total_loss))),
                  ', accuracy: %.4f' % (sum(total_acc) / float(len(total_acc))),
                  ', %.5s seconds' % (time.time() - start_time),
                  ', dev loss: %.4f' % (dev_loss),
                  ', dev accuracy: %.4f' % (dev_acc),
                  )

        if self.monitor == 'val_acc':
            if self.glob_best_acc < best_acc:
                self.glob_best_acc = best_acc
                self.glob_best_loss = best_loss
                self.glob_best_params = args
                torch.save(model, GLOB_BEST_MODEL_PATH)

        elif self.monitor == 'val_loss':
            if self.glob_best_loss > best_loss:
                self.glob_best_loss = best_loss
                self.glob_best_acc = best_acc
                self.glob_best_params = args
                torch.save(model, GLOB_BEST_MODEL_PATH)
        else:
            raise Exception('Wrong monitor parameter...')

        print("Current params:", args)
        print("best loss of current params:", best_loss, ', acc:', best_acc)
        print("params of glob best loss:", self.glob_best_params)
        print("glob best loss:", self.glob_best_loss, ', acc:', self.glob_best_acc)
        print("*" * 80)

def main():

    temp_extractor = TempOptimizer(['Event-Timex', 'Timex-Event'], 'val_loss')
    temp_extractor.optimize_model(max_evals=100)


if __name__ == '__main__':
    main()














