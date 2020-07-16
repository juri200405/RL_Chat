import argparse
import json
from queue import Queue

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

import torch_optimizer

import sentencepiece as spm

import encoder_decoder
from dataloader import get_dataloader

class Q_network(nn.Module):
    def __init__(self, obs_size=1024, action_size=1024, mid_size=1024):
        super(Q_network, self).__init__()
        self.fc1 = nn.Linear(obs_size+action_size, mid_size)
        self.fc2 = nn.Linear(mid_size, 1)

    def forward(self, action, obs):
        inp = torch.cat((action, obs), dim=-1)
        q = F.relu(self.fc1(inp))
        q = self.fc2(q)
        return q

class Policy_network(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, obs_size=1024, mid_size=1024, output_size=1024):
        super(Policy_network, self).__init__()
        self.fc1 = nn.Linear(obs_size, mid_size)
        self.fc2 = nn.Linear(mid_size, output_size*2)

    def forward(self, obs):
        policy = F.relu(self.fc1(obs))
        policy = self.fc2(policy)

        mean, log_std = torch.chunk(policy, 2, dim=-1)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        std = torch.exp(log_std)

        return mean, std

    def sample(self, obs):
        mean, std = self.forward(obs)
        m = MultivariateNormal(mean, torch.diag_embed(std))
        action = m.rsample()
        log_prob = m.log_prob(action)

        return action, mean, log_prob

    def get_log_prob(self, obs, action):
        mean, std = self.forward(obs)
        m = MultivariateNormal(mean, torch.diag_embed(std))
        log_prob = m.log_prob(action)

        return action, mean, log_prob

class Agent:
    def __init__(self, device, writer):
        self.device = device

        self.gru = nn.GRU(input_size=1024, hidden_size=1024)
        self.policy = Policy_network()
        self.qf1 = Q_network()
        self.qf2 = Q_network()
        self.target_qf1 = Q_network()
        self.target_qf2 = Q_network()
        self.log_alpha = torch.tensor([0.1], requires_grad=True, device=device)

        self.target_entropy = -1024
        self.discount = 0.99
        self.tau = 5e-3

        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        self.to(device)

        self.target_qf1.eval()
        self.target_qf2.eval()

        self.gru_opt = optim.Adam(self.gru.parameters())
        self.policy_opt = optim.Adam(self.policy.parameters())
        self.qf1_opt = optim.Adam(self.qf1.parameters())
        self.qf2_opt = optim.Adam(self.qf2.parameters())
        self.alpha_opt = optim.Adam([self.log_alpha])

        self.writer = writer

    def state_dict(self):
        return dict(
                gru = self.gru.state_dict(),
                policy = self.policy.state_dict(),
                qf1 = self.qf1.state_dict(),
                qf2 = self.qf2.state_dict(),
                target_qf1 = self.target_qf1.state_dict(),
                target_qf2 = self.target_qf2.state_dict(),
                log_alpha = self.log_alpha.item()
                )

    def load_state_dict(self, state):
        self.gru.load_state_dict(state["gru"])
        self.policy.load_state_dict(state["policy"])
        self.qf1.load_state_dict(state["qf1"])
        self.qf2.load_state_dict(state["qf2"])
        self.target_qf1.load_state_dict(state["target_qf1"])
        self.target_qf2.load_state_dict(state["target_qf2"])
        self.log_alpha = torch.tensor(state["log_alpha"], requires_grad=True, device=self.device)

        self.to(self.device)

    def to(self, device):
        self.gru.to(device)
        self.policy.to(device)
        self.qf1.to(device)
        self.qf2.to(device)
        self.target_qf1.to(device)
        self.target_qf2.to(device)

    def learn(self, i, state, hidden, action, reward, next_state, next_hidden, is_final):
        state = state.to(self.device)
        hidden = hidden.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        next_hidden = next_hidden.to(self.device)
        is_final = is_final.to(self.device)

        obs, _ = self.gru(state, hidden)
        obs_q1 = obs.detach().requires_grad_()
        obs_q2 = obs.detach().requires_grad_()
        obs_policy = obs.detach().requires_grad_()

        q1 = self.qf1(action, obs_q1)
        q2 = self.qf2(action, obs_q2)
        _, _, log_prob = self.policy.get_log_prob(obs_policy, action)
        with torch.no_grad():
            min_q = torch.min(q1, q2)
            next_obs, _ = self.gru(next_state, next_hidden)
            next_action, _, next_log_prob = self.policy.sample(next_obs)
            target_q = torch.min(self.target_qf1(next_action, next_obs), self.target_qf2(next_action, next_obs))
            alpha = torch.exp(self.log_alpha)
            q_loss_scale = min_q - (reward + (1 - is_final) * self.discount * (target_q - alpha * next_log_prob)).detach()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        policy_loss = (alpha * log_prob - min_q).mean()
        qf1_loss = (q1 * q_loss_scale).mean()
        qf2_loss = (q2 * q_loss_scale).mean()

        # 各パラメータの更新
        self.gru_opt.zero_grad()
        self.policy_opt.zero_grad()
        self.qf1_opt.zero_grad()
        self.qf2_opt.zero_grad()
        self.alpha_opt.zero_grad()

        policy_loss.backward()
        qf1_loss.backward()
        qf2_loss.backward()
        alpha_loss.backward()
        obs.backward(obs_q1.grad + obs_q2.grad + obs_policy.grad)

        self.gru_opt.step()
        self.policy_opt.step()
        self.qf1_opt.step()
        self.qf2_opt.step()
        self.alpha_opt.step()

        # target_qfの更新
        for t_p, p in zip(self.target_qf1.parameters(), self.qf1.parameters()):
            t_p.data.copy_(t_p.data * (1.0-self.tau) + p.data * self.tau)
        for t_p, p in zip(self.target_qf2.parameters(), self.qf2.parameters()):
            t_p.data.copy_(t_p.data * (1.0-self.tau) + p.data * self.tau)

        if self.writer is not None:
            print("writer update")
            self.writer.add_scalar("alpha_loss", alpha_loss.item(), i)
            self.writer.add_scalar("policy_loss", policy_loss.item(), i)
            self.writer.add_scalar("qf1_loss", qf1_loss.item(), i)
            self.writer.add_scalar("qf2_loss", qf2_loss.item(), i)

    def train(self):
        self.gru.train()
        self.policy.train()
        self.qf1.train()
        self.qf2.train()
        
    def eval(self):
        self.gru.eval()
        self.policy.eval()
        self.qf1.eval()
        self.qf2.eval()


class Chat_Module:
    def __init__(self, spm_model, encoder, decoder, encoder_device="cpu", decoder_device="cpu", learning_agent_device="cpu", chat_agent_device="cpu", batch_size=64, writer=None):
        self.sp = spm_model

        # encoderのBERT内に組み込まれている BertEmbeddings をdecoderで使うため、GPUに送る順番は decoder->encoder
        self.decoder = decoder.to(decoder_device)
        self.encoder = encoder.to(encoder_device)
        self.decoder.eval()
        self.encoder.eval()

        self.encoder_device = encoder_device
        self.decoder_device = decoder_device

        self.learning_agent = Agent(learning_agent_device, writer=writer)
        self.chat_agent = Agent(chat_agent_device, writer=writer)
        self.last_param = self.learning_agent.state_dict() 

        self.sample_queue = Queue()
        self.batch_size = batch_size

    def make_utt(self, usr_utt, hidden):
        self.chat_agent.eval()
        with torch.no_grad():
            state = self.encode(usr_utt) # state = (1, batch(1), 1024)
            obs, next_hidden = self.chat_agent.gru(state.to(self.chat_agent.device), hidden.to(self.chat_agent.device)) # obs = (1, batch(1), 1024), hidden = (1, batch(1), 1024)
            action, _, _ = self.chat_agent.policy.sample(obs)

            sys_utt = self.decode(action.to(self.decoder_device), 256)
        return sys_utt, next_hidden

    def initial_hidden(self):
        return torch.zeros(1,1,1024)

    def decode(self, memory, max_length):
        with torch.no_grad():
            tgt = torch.full((1, max_length), 0, dtype=torch.long, device=self.encoder_device)
            tgt_key_padding_mask = torch.full((1, max_length), True, dtype=torch.bool, device=self.decoder_device)
            t_word = 1 # <s>
            ids = []
            for t in range(max_length):
                tgt[0][t] = t_word
                tgt_key_padding_mask[0][t] = False
                out = self.decoder(tgt, memory, tgt_padding_mask=tgt_key_padding_mask)
                _, topi = out.topk(1)
                next_word = topi[t].item()
                ids.append(next_word)
                if next_word == 2: # </s>
                    break
                else:
                    t_word = next_word
            print(ids)
        return self.sp.decode(ids)

    def update_param(self, sample):
        self.sample_queue.put(sample)
        self.chat_agent.load_state_dict(self.last_param)

    def update_param_from_file(self, path):
        pass

    def encode(self, utt):
        with torch.no_grad():
            input_tensor = torch.LongTensor([1] + self.sp.encode(utt) + [2]).unsqueeze(0).to(self.encoder_device)
            state = self.encoder(input_tensor) # state = (1, batch(1), 1024)
        return state

    def make_batch(self, sample):
        memory = []
        for dialog in sample:
            usr = None
            sys = None
            hidden = self.initial_hidden()
            for item in dialog:
                next_usr = item["usr_utt"]
                next_sys = item["sys_utt"]
                if usr is not None:
                    state = self.encode(usr)
                    with torch.no_grad():
                        _, next_hidden = self.learning_agent.gru(state.to(self.learning_agent.device), hidden.to(self.learning_agent.device))
                    action = self.encode(sys)
                    reward = torch.tensor([[1]])
                    is_final = torch.tensor([[0]])
                    next_state = self.encode(next_usr)
                    memory.append(
                            dict(
                                state=state.cpu(),
                                hidden=hidden.cpu(),
                                action=action.cpu(),
                                reward=reward.cpu(),
                                next_state=next_state.cpu(),
                                next_hidden=next_hidden.cpu(),
                                is_final=is_final.cpu()
                                )
                            )
                    hidden = next_hidden
                usr = next_usr
                sys = next_sys

            state = self.encode(usr)
            action = self.encode(sys)
            reward = torch.tensor([[0]])
            is_final = torch.tensor([[1]])
            memory.append(
                    dict(
                        state=state.cpu(),
                        hidden=hidden.cpu(),
                        action=action.cpu(),
                        reward=reward.cpu(),
                        next_state=torch.zeros_like(state).cpu(),
                        next_hidden=torch.zeros_like(hidden).cpu(),
                        is_final=is_final.cpu()
                        )
                    )
        return memory

    def learn(self):
        learn_num = 0
        while True:
            sample = self.sample_queue.get()
            print("learning")
            self.learning_agent.train()

            itr = get_dataloader(self.make_batch(sample), self.batch_size)
            for batch in itr:
                self.learning_agent.learn(learn_num, *batch)
                learn_num += 1

            self.last_param = self.learning_agent.state_dict() 

            self.sample_queue.task_done()
