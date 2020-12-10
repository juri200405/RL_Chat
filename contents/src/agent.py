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
    def __init__(self, n_latent, obs_size, device, writer):
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
        obs_policy = obs.detach().requires_grad_()
        obs_q1 = obs.detach().requires_grad_()
        obs_q2 = obs.detach().requires_grad_()

        new_obs_action, _, log_prob = self.policy.sample(obs_policy)
        log_prob = log_prob.unsqueeze(-1)
        q1 = self.qf1(action, obs_q1)
        q2 = self.qf2(action, obs_q2)

        with torch.no_grad():
            min_q = torch.min(self.qf1(new_obs_action, obs), self.qf2(new_obs_action, obs))
            next_obs, _ = self.gru(next_state, next_hidden)
            next_action, _, next_log_prob = self.policy.sample(next_obs)
            next_log_prob = next_log_prob.unsqueeze(-1)
            alpha = torch.exp(self.log_alpha)
            target_q = torch.min(self.target_qf1(next_action, next_obs), self.target_qf2(next_action, next_obs)) - alpha*next_log_prob
            q_target = (reward + (1 - is_final) * self.discount * target_q).detach()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        policy_loss = (alpha * log_prob - min_q.detach()).mean()
        qf1_loss = self.qf_criterion(q1, q_target)
        qf2_loss = self.qf_criterion(q2, q_target)

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
    def __init__(
            self,
            spm_model,
            encoder,
            decoder,
            encoder_device="cpu",
            decoder_device="cpu",
            learning_agent_device="cpu",
            chat_agent_device="cpu",
            batch_size=64,
            n_latent=1024,
            max_len=128,
            obs_size=1024,
            num_beams=5,
            writer=None
            ):
        self.sp = spm_model

        self.encoder = encoder.to(encoder_device)
        self.decoder = decoder.to(decoder_device)
        self.decoder.eval()
        self.encoder.eval()

        self.encoder_device = encoder_device
        self.decoder_device = decoder_device

        self.learning_agent = Agent(n_latent, obs_size, learning_agent_device, writer=writer)
        self.chat_agent = Agent(n_latent, obs_size, chat_agent_device, writer=writer)
        self.last_param = self.learning_agent.state_dict() 

        self.sample_queue = Queue()
        self.batch_size = batch_size
        self.max_len = max_len
        self.obs_size = obs_size
        self.num_beams=num_beams

    def make_utt(self, usr_utt, hidden):
        self.chat_agent.eval()
        with torch.no_grad():
            state = self.encode(usr_utt) # state = (1, batch(1), n_latent)
            obs, next_hidden = self.chat_agent.gru(state.to(self.chat_agent.device), hidden.to(self.chat_agent.device)) # obs = (1, batch(1), n_latent), hidden = (1, batch(1), 1024)
            action, _, _ = self.chat_agent.policy.sample(obs)

            sys_utt = self.decode(action.to(self.decoder_device))
        return sys_utt[0], next_hidden

    def initial_hidden(self):
        return torch.zeros(1,1,self.obs_size)

    def encode(self, utt):
        with torch.no_grad():
            input_s = torch.tensor([1] + self.sp.encode(utt) + [2], dtype=torch.long, device=self.encoder_device).unsqueeze(0)
            inp_mask = torch.tensor([[False]*input_s.shape[1] + [True]*(self.max_len - input_s.shape[1])], device=self.encoder_device)
            pad = torch.full((1, self.max_len - input_s.shape[1]), 3, dtype=torch.long, device=self.encoder_device)
            input_s = torch.cat((input_s, pad), dim=1)
            state = self.encoder(input_s, attention_mask=inp_mask) # state = (batch(1), n_latent)
        return state.unsqueeze(0)

    def decode(self, memory):
        if self.num_beams < 2:
            return self.greedy_decode(memory)
        else:
            return self.beam_decode(memory)

    def greedy_decode(self, memory):
        with torch.no_grad():
            tgt = torch.full((memory.shape[0], 1), 1, dtype=torch.long, device=self.decoder_device)  # <s>
            unfinish = torch.ones(memory.shape[0], 1, dtype=torch.long, device=self.decoder_device)
            memory = memory.to(self.decoder_device)
            while tgt.shape[1] < self.max_len:
                out = self.decoder(tgt, memory)
                _, topi = out.transpose(0,1).topk(1)
                next_word = topi[:,-1]
                next_word = next_word*unfinish + (3)*(1-unfinish)
                tgt = torch.cat((tgt, next_word), dim=-1)
                unfinish = unfinish * (~(next_word == 2)).long()
                if unfinish.max() == 0: # </s>
                    break
        return self.sp.decode(tgt.cpu().tolist())

    def beam_decode(self, memory):
        num_beams = self.num_beams
        batch_size = memory.shape[0]
        vocab_size = len(self.sp)
        beam_scorer = transformers.BeamSearchScorer(batch_size=batch_size, max_length=self.max_len, num_beams=num_beams, device=self.decoder_device)
        with torch.no_grad():
            memory_list = torch.split(memory, 1)
            memory_list = [torch.cat([item]*num_beams) for item in memory_list]
            memory = torch.cat(memory_list)
            memory = memory.to(self.decoder_device)

            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=self.decoder_device)
            beam_scores[:,1:] = -1e9
            beam_scores = beam_scores.view((batch_size * num_beams,))

            tgt = torch.full((memory.shape[0], 1), 1, dtype=torch.long, device=self.decoder_device)  # <s>
            while tgt.shape[1] < self.max_len:
                out = self.decoder(tgt, memory) # (seq, batch, n_vocab)
                out = out.transpose(0,1)[:, -1, :]

                next_token_scores = torch.nn.functional.log_softmax(out, dim=-1)
                next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

                next_token_scores, next_tokens = torch.topk(next_token_scores, 2*num_beams, dim=1, largest=True, sorted=True)
                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size

                beam_outputs = beam_scorer.process(tgt, next_token_scores, next_tokens, next_indices, pad_token_id=3, eos_token_id=2)
                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                tgt = torch.cat((tgt[beam_idx, :], beam_next_tokens.unsqueeze(-1)), dim=-1)

                if beam_scorer.is_done:
                    break
            decoded = beam_scorer.finalize(tgt, beam_scores, next_tokens, next_indices, pad_token_id=3, eos_token_id=2)
        return self.sp.decode(decoded.cpu().tolist())

    def update_param(self, sample):
        self.sample_queue.put(sample)
        self.chat_agent.load_state_dict(self.last_param)

    def update_param_from_file(self, path):
        pass

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
            self.learning_agent.train()

            itr = get_dataloader(self.make_batch(sample), self.batch_size)
            for batch in itr:
                self.learning_agent.learn(learn_num, *batch)
                learn_num += 1

            self.last_param = self.learning_agent.state_dict() 

            self.sample_queue.task_done()
