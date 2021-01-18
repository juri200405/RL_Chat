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
    def __init__(self, activation, obs_size=1024, action_size=1024, mid_size=1024):
        super(Q_network, self).__init__()
        self.fc1 = nn.Linear(obs_size+action_size, mid_size)
        self.fc2 = nn.Linear(mid_size, 1)
        self.relu = torch.nn.LeakyReLU()
        self.activation = activation

    def forward(self, action, obs):
        # action : (batch_size, action_size)
        # obs : (batch_size, obs_size)

        inp = torch.cat((action, obs), dim=-1)
        h = self.relu(self.fc1(inp))
        q = self.fc2(h)
        # q : (batch_size, 1)
        return self.activation(q.squeeze()) # (batch_size)

class Policy_network(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, obs_size=1024, mid_size=1024, output_size=1024):
        super(Policy_network, self).__init__()
        self.fc = nn.Linear(obs_size, mid_size)
        self.mid2mean = torch.nn.Linear(mid_size, output_size)
        self.mid2logv = torch.nn.Linear(mid_size, output_size)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, obs):
        policy = self.relu(self.fc(obs))
        mean = self.mid2mean(policy)
        log_std = self.mid2logv(policy)

        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        std = log_std.exp()

        return mean, std # (batch_size, action_size)

    def sample(self, obs):
        # obs : (batch_size, state_size)
        mean, std = self.forward(obs)
        m = MultivariateNormal(mean, torch.diag_embed(std))
        xs = m.rsample()
        action = self.tanh(xs)
        log_prob = self.log_prob(xs, m)

        return action, log_prob # (batch_size, action_size)

    def get_log_prob(self, obs, action):
        mean, std = self.forward(obs)
        m = MultivariateNormal(mean, torch.diag_embed(std))
        val = torch.clamp(action, -0.999999, 0.999999)
        val = torch.log(1+val) / 2 - torch.log(1-val) / 2
        log_prob = self.log_prob(val, m)

        return action, log_prob

    def log_prob(self, val, m):
        log_prob = m.log_prob(val)
        correction = -2.0 * (
                torch.tensor([2.0], device=val.device).log()
                - val
                - torch.nn.functional.softplus(-2.0 * val)
                ).sum(dim=1)
        return log_prob + correction

class Agent:
    def __init__(
            self,
            activation,
            n_latent,
            mid_size,
            device,
            lr=1e-3,
            target_entropy=-128,
            discount=0.99,
            tau=5e-3,
            initial_log_alpha=0.1,
            ):
        self.device = device

        self.policy = Policy_network(obs_size=n_latent, output_size=n_latent, mid_size=mid_size)
        self.qf1 = Q_network(activation, obs_size=n_latent, action_size=n_latent, mid_size=mid_size)
        self.qf2 = Q_network(activation, obs_size=n_latent, action_size=n_latent, mid_size=mid_size)
        self.target_qf1 = Q_network(activation, obs_size=n_latent, action_size=n_latent, mid_size=mid_size)
        self.target_qf2 = Q_network(activation, obs_size=n_latent, action_size=n_latent, mid_size=mid_size)
        self.log_alpha = torch.tensor([initial_log_alpha], requires_grad=True, device=device)

        self.qf_criterion = nn.MSELoss()

        self.target_entropy = target_entropy
        self.discount = discount
        self.tau = tau

        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        self.to(device)

        self.target_qf1.eval()
        self.target_qf2.eval()

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.qf1_opt = optim.Adam(self.qf1.parameters(), lr=lr)
        self.qf2_opt = optim.Adam(self.qf2.parameters(), lr=lr)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

    def state_dict(self):
        return dict(
                policy = self.policy.state_dict(),
                qf1 = self.qf1.state_dict(),
                qf2 = self.qf2.state_dict(),
                target_qf1 = self.target_qf1.state_dict(),
                target_qf2 = self.target_qf2.state_dict(),
                log_alpha = self.log_alpha.item()
                )

    def load_state_dict(self, state):
        self.policy.load_state_dict(state["policy"])
        self.qf1.load_state_dict(state["qf1"])
        self.qf2.load_state_dict(state["qf2"])
        self.target_qf1.load_state_dict(state["target_qf1"])
        self.target_qf2.load_state_dict(state["target_qf2"])
        self.log_alpha = torch.tensor(state["log_alpha"], requires_grad=True, device=self.device)

        self.to(self.device)

    def to(self, device):
        self.policy.to(device)
        self.qf1.to(device)
        self.qf2.to(device)
        self.target_qf1.to(device)
        self.target_qf2.to(device)

    def learn(self, state, action, reward, next_state, is_final, graph=False):
        state = state.to(self.device)               # (batch_size, n_latent)
        action = action.to(self.device)             # (batch_size, n_latent)
        reward = reward.to(self.device)             # (batch_size)
        next_state = next_state.to(self.device)     # (batch_size, n_latent)
        is_final = is_final.to(self.device)         # (batch_size)

        new_obs_action, log_prob = self.policy.sample(state)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        alpha = torch.exp(self.log_alpha)
        # alpha_loss = (-alpha * (log_prob + self.target_entropy).detach()).mean()

        min_q = torch.min(self.qf1(new_obs_action, state), self.qf2(new_obs_action, state))
        policy_loss = (alpha * log_prob - min_q).mean()
        # policy_loss = (alpha.detach() * log_prob - min_q).mean()

        q1 = self.qf1(action, state)
        q2 = self.qf2(action, state)

        next_action, next_log_prob = self.policy.sample(next_state)
        target_q = torch.min(self.target_qf1(next_action, next_state), self.target_qf2(next_action, next_state))

        q_target = (reward + (1 - is_final) * self.discount * (target_q - alpha*next_log_prob)).detach()

        qf1_loss = self.qf_criterion(q1, q_target.detach())
        qf2_loss = self.qf_criterion(q2, q_target.detach())

        # 各パラメータの更新
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.policy_opt.zero_grad()
        self.qf1_opt.zero_grad()
        self.qf2_opt.zero_grad()

        policy_loss.backward()
        qf1_loss.backward()
        qf2_loss.backward()

        self.policy_opt.step()
        self.qf1_opt.step()
        self.qf2_opt.step()

        # target_qfの更新
        for t_p, p in zip(self.target_qf1.parameters(), self.qf1.parameters()):
            t_p.data.copy_(t_p.data * (1.0-self.tau) + p.data * self.tau)
        for t_p, p in zip(self.target_qf2.parameters(), self.qf2.parameters()):
            t_p.data.copy_(t_p.data * (1.0-self.tau) + p.data * self.tau)

        out_dict = {
                "loss/alpha_loss": alpha_loss.item(),
                "loss/policy_loss": policy_loss.item(),
                "loss/qf1_loss": qf1_loss.item(),
                "loss/qf2_loss": qf2_loss.item(),
                "debug/alpha": alpha.item(),
                "debug/log_prob": log_prob.mean().item(),
                "debug/min_q": min_q.mean().item(),
                "debug/q1": q1.mean().item(),
                "debug/q2": q2.mean().item(),
                "debug/reward": reward.mean().item()
                }

        if graph:
            losses = {"alpha_loss":alpha_loss, "policy_loss":policy_loss, "qf1_loss":qf1_loss, "qf2_loss":qf2_loss}
        else:
            losses = None

        return out_dict, losses

    def train(self):
        self.policy.train()
        self.qf1.train()
        self.qf2.train()
        
    def eval(self):
        self.policy.eval()
        self.qf1.eval()
        self.qf2.eval()

    def act(self, state):
        action, _ = self.policy.sample(state)
        return action
