import argparse
import json
from queue import Queue

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

import sentencepiece as spm
import transformers

import encoder_decoder
from dataloader import get_dataloader

class Q_network(nn.Module):
    def __init__(self, obs_size=1024, action_size=1024, mid_size=1024):
        super(Q_network, self).__init__()
        self.fc1 = nn.Linear(obs_size+action_size, mid_size)
        self.fc2 = nn.Linear(mid_size, 1, bias=False)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, action, obs):
        # action : (batch_size, action_size)
        # obs : (batch_size, obs_size)

        inp = torch.cat((action, obs), dim=-1)
        h = self.relu(self.fc1(inp))
        q = self.fc2(h)
        # q : (batch_size, 1)
        return self.activation(q.squeeze()) # (batch_size)

    def activation(self, x):
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        root = torch.sqrt(abs_x)
        return torch.where(abs_x < 1, x, sign*root)

class Policy_network(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, obs_size=1024, mid_size=1024, output_size=1024):
        super(Policy_network, self).__init__()
        self.fc = nn.Linear(obs_size, mid_size)
        self.mid2mean = torch.nn.Linear(mid_size, output_size)
        self.mid2logv = torch.nn.Linear(mid_size, output_size)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, obs):
        policy = self.relu(self.fc(obs))
        mean = self.mid2mean(policy)
        log_std = self.mid2logv(policy)

        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        std = torch.exp(log_std)

        return mean, std # (batch_size, action_size)

    def sample(self, obs):
        # obs : (batch_size, state_size)
        mean, std = self.forward(obs)
        m = MultivariateNormal(mean, torch.diag_embed(std))
        action = m.rsample()
        log_prob = m.log_prob(action)

        return action, mean, log_prob # (batch_size, action_size)

    def get_log_prob(self, obs, action):
        mean, std = self.forward(obs)
        m = MultivariateNormal(mean, torch.diag_embed(std))
        log_prob = m.log_prob(action)

        return action, mean, log_prob

class Agent:
    def __init__(self, n_latent, obs_size, device, lr=1e-3):
        self.device = device

        self.gru = nn.GRU(input_size=n_latent, hidden_size=obs_size)
        self.policy = Policy_network(obs_size=obs_size, output_size=n_latent)
        self.qf1 = Q_network(obs_size=obs_size, action_size=n_latent)
        self.qf2 = Q_network(obs_size=obs_size, action_size=n_latent)
        self.target_qf1 = Q_network(obs_size=obs_size, action_size=n_latent)
        self.target_qf2 = Q_network(obs_size=obs_size, action_size=n_latent)
        self.log_alpha = torch.tensor([0.1], requires_grad=True, device=device)

        self.qf_criterion = nn.MSELoss()

        self.target_entropy = -1024
        self.discount = 0.0
        self.tau = 5e-3

        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        self.to(device)

        self.target_qf1.eval()
        self.target_qf2.eval()

        self.gru_opt = optim.Adam(self.gru.parameters(), lr=lr)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.qf1_opt = optim.Adam(self.qf1.parameters(), lr=lr)
        self.qf2_opt = optim.Adam(self.qf2.parameters(), lr=lr)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

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

    def learn(self, state, hidden, action, reward, next_state, next_hidden, is_final):
        state = state.to(self.device)               # (1, batch_size, n_latent)
        hidden = hidden.to(self.device)             # (1, batch_size, obs_size)
        action = action.to(self.device)             # (batch_size, n_latent)
        reward = reward.to(self.device)             # (batch_size)
        next_state = next_state.to(self.device)     # (1, batch_size, n_latent)
        next_hidden = next_hidden.to(self.device)   # (1, batch_size, obs_size)
        is_final = is_final.to(self.device)         # (batch_size)

        obs, _ = self.gru(state, hidden)
        obs = obs.squeeze(0)
        # obs_policy = obs.detach().requires_grad_()
        # obs_q1 = obs.detach().requires_grad_()
        # obs_q2 = obs.detach().requires_grad_()

        # new_obs_action, _, log_prob = self.policy.sample(obs_policy)
        new_obs_action, _, log_prob = self.policy.sample(obs)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        alpha = torch.exp(self.log_alpha)

        # min_q = torch.min(self.qf1(new_obs_action, obs_policy), self.qf2(new_obs_action, obs_policy))
        min_q = torch.min(self.qf1(new_obs_action, obs), self.qf2(new_obs_action, obs))
        policy_loss = (alpha * log_prob - min_q).mean()

        # q1 = self.qf1(action, obs_q1)
        # q2 = self.qf2(action, obs_q2)
        q1 = self.qf1(action, obs)
        q2 = self.qf2(action, obs)

        next_obs, _ = self.gru(next_state, next_hidden)
        next_obs = next_obs.squeeze(0)
        next_action, _, next_log_prob = self.policy.sample(next_obs)
        target_q = torch.min(self.target_qf1(next_action, next_obs), self.target_qf2(next_action, next_obs))
        q_target = (reward + (1 - is_final) * self.discount * (target_q - alpha*next_log_prob)).detach()

        qf1_loss = self.qf_criterion(q1, q_target)
        qf2_loss = self.qf_criterion(q2, q_target)

        # 各パラメータの更新
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.gru_opt.zero_grad()
        self.policy_opt.zero_grad()
        self.qf1_opt.zero_grad()
        self.qf2_opt.zero_grad()

        loss = policy_loss + qf1_loss + qf2_loss
        loss.backward()
        # policy_loss.backward()
        # qf1_loss.backward()
        # qf2_loss.backward()
        # obs.backward(obs_policy.grad + obs_q2.grad + obs_policy.grad)

        self.gru_opt.step()
        self.policy_opt.step()
        self.qf1_opt.step()
        self.qf2_opt.step()

        # target_qfの更新
        for t_p, p in zip(self.target_qf1.parameters(), self.qf1.parameters()):
            t_p.data.copy_(t_p.data * (1.0-self.tau) + p.data * self.tau)
        for t_p, p in zip(self.target_qf2.parameters(), self.qf2.parameters()):
            t_p.data.copy_(t_p.data * (1.0-self.tau) + p.data * self.tau)

        return {
                "loss/alpha_loss": alpha_loss.item(),
                "loss/policy_loss": policy_loss.item(),
                "loss/qf1_loss": qf1_loss.item(),
                "loss/qf2_loss": qf2_loss.item(),
                "debug/log_alpha": self.log_alpha.item(),
                "debug/alpha": alpha.item(),
                "debug/log_prob": log_prob.mean().item(),
                "debug/min_q": min_q.mean().item(),
                "debug/q1": q1.mean().item(),
                "debug/q2": q2.mean().item(),
                "debug/reward": reward.mean().item()
                }

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

    def act(self, state, hidden):
        obs, next_hidden = self.gru(state, hidden)
        # obs : (batch_size(1), obs_size)
        # hidden : (1, batch_size(1), obs_size)
        action, _, _ = self.policy.sample(obs.squeeze(0))
        return action, next_hidden


class Chat_Module:
    def __init__(
            self,
            learning_agent_device="cpu",
            chat_agent_device="cpu",
            batch_size=64,
            n_latent=1024,
            max_len=128,
            obs_size=1024,
            num_beams=5,
            writer=None
            ):

        self.learning_agent = Agent(n_latent, obs_size, learning_agent_device, writer=writer)
        self.chat_agent = Agent(n_latent, obs_size, chat_agent_device, writer=writer)
        self.last_param = self.learning_agent.state_dict() 

        self.sample_queue = Queue()
        self.batch_size = batch_size
        self.max_len = max_len
        self.obs_size = obs_size

    def act(self, state, hidden):
        self.chat_agent.eval()
        with torch.no_grad():
            action, next_hidden = self.chat_agent.act(state, hidden)

        return action.cpu(), next_hidden.cpu()

    def initial_hidden(self):
        return torch.zeros(1,1,self.obs_size)

    def update_param(self, sample):
        self.sample_queue.put(sample)
        self.chat_agent.load_state_dict(self.last_param)

    def update_param_from_file(self, path):
        pass

    def make_batch(self, sample):
        memory = []
        for dialog in sample:
            for i in range(len(dialog)):
                state = dialog[i]["state"]
                hidden = dialog[i]["hidden"]
                action = dialog[i]["action"]
                score = dialog[i]["score"]
                reward = torch.tensor([[score]])
                if i+1 < len(dialog):
                    next_state = dialog[i+1]["state"]
                    next_hidden = dialog[i+1]["hidden"]
                    is_final = torch.tensor([[0]])
                else:
                    next_state = torch.zeros_like(state)
                    next_hidden = torch.zeros_like(hidden)
                    is_final = torch.tensor([[1]])
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
        return memory

    def learn(self):
        learn_num = 0
        while True:
            sample = self.sample_queue.get()
            self.learning_agent.train()

            itr = get_dataloader(self.make_batch(sample), self.batch_size)
            for batch in itr:
                self.learning_agent.learn(learn_num, *batch)
                learn_num += 1

            self.last_param = self.learning_agent.state_dict() 

            self.sample_queue.task_done()
