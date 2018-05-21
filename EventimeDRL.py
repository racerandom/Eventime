import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from IPython.display import clear_output
import numpy as np
import random
import time as timer
import pdb
import gensim
from itertools import count
from torch.distributions import Categorical

import eventime as Tlink


GAMMA = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dct_inputs = torch.randint(0, 100, (50, 1, 3), dtype=torch.long, device=device)
time_inputs = torch.randint(0, 100, (50, 1, 2, 3), dtype=torch.long, device=device)

targets = torch.randint(0, 2, (50, 3), dtype=torch.long, device=device)

# dct_inputs = torch.LongTensor(500, 1, 3).random_(0, 100)
# time_inputs = torch.LongTensor(500, 1, 2, 3).random_(0, 100)
#
# targets = torch.Tensor(500, 3).random_(0, 2)

BATCH_SIZE = 1
# EPOCH_NUM = 100

print(dct_inputs.size(), time_inputs.size(), targets.size())

dataset = Tlink.MultipleDatasets(dct_inputs, time_inputs, targets)

loader = Data.DataLoader(
    dataset = dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 1,
)

EMBEDDING_DIM = 64
DCT_HIDDEN_DIM = 60
TIME_HIDDEN_DIM = 50
VOCAB_SIZE = 100
ACTION_SIZE = 8
EPOCH_NUM = 50

policy = Tlink.DqnInferrer(EMBEDDING_DIM, DCT_HIDDEN_DIM, TIME_HIDDEN_DIM, VOCAB_SIZE, ACTION_SIZE, BATCH_SIZE).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state_i, dpath_input):
    probs = policy(state_i, dpath_input)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():

    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + GAMMA * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards, device=device)
    # print(rewards, eps, rewards.mean(), rewards.std())
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps) # reward normalization
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        # print(log_prob, reward, -log_prob * reward)
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    # print('finish:', policy.rewards, policy.saved_log_probs, torch.cat(policy_loss))
    policy_loss = torch.cat(policy_loss).sum()

    policy_loss.backward(retain_graph=True)
    optimizer.step()

    del policy.rewards[:]
    del policy.saved_log_probs[:]
    return policy_loss


def step(state_i, action, anchor, target):

    update_strategies = torch.tensor([[0, 0, 0],
                                      [1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1],
                                      [1, 1, 0],
                                      [0, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.long, device=device)
    timex = torch.tensor([0, 0, 0], dtype=torch.long, device=device) \
        if state_i == 0 else torch.tensor([1, 1, 1], dtype=torch.long, device=device)
    anchor = update_strategies[action] * timex \
             + (torch.ones_like(update_strategies[action], dtype=torch.long, device=device) - update_strategies[action]) \
             * anchor
    # print(anchor, target)
    # print(torch.eq(anchor, target).sum(), anchor.numel())
    reward = torch.div(torch.eq(anchor, target).sum().float(), anchor.numel())
    state_i += 1
    return state_i, reward, anchor


def main():

    for epoch in range(EPOCH_NUM):
        start_time = timer.time()
        total_loss = torch.Tensor([0])
        total_reward = torch.Tensor([0])
        for episode, (dct_input, time_inputs, target) in enumerate(loader):

            ## dct state
            state_i = 0
            dct_action = select_action(state_i, dct_input)
            anchor = torch.tensor([-1, -1, -1], dtype=torch.long, device=device)
            state_i, reward, anchor = step(state_i, dct_action, anchor, target)
            # print('dct action:', dct_action, anchor, reward)
            policy.rewards.append(reward)

            ## time states
            for timex in range(time_inputs.size()[1]):
                time_input = time_inputs[:, timex, :, :]
                time_action = select_action(state_i, time_input)
                state_i, reward, anchor = step(state_i, time_action, anchor, target)
                # print('time action:', time_action, anchor, reward)
                policy.rewards.append(reward)
            total_reward += reward
            total_loss += finish_episode()
        print('epoch:', epoch, ', loss: %.4f' % (total_loss.item()/50), ', reward: %.4f' % (total_reward.item() / 50), ", %.4s seconds" % (timer.time() - start_time))


if __name__ == '__main__':
    main()
