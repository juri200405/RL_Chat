import argparse
from pathlib import Path
import json
import re

import torch
from torch.utils.tensorboard import SummaryWriter

import sentencepiece as spm

from vae_check import VAE_tester
from config import Config

class Q_network(torch.nn.Module):
    def __init__(self, action_size, mid_size):
        super(Q_network, self).__init__()
        self.fc1 = torch.nn.Linear(action_size, mid_size)
        self.fc2 = torch.nn.Linear(mid_size, 1)
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, action):
        # action : (batch_size, action_size)

        h = self.relu(self.fc1(action))
        # h : (batch_size, mid_size)

        q = self.sigmoid(self.fc2(h))
        # q : (batch_size, 1)
        return  q.squeeze() # (batch_size)


class Policy_network(torch.nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, state_size, action_size, mid_size):
        super(Policy_network, self).__init__()
        self.fc = torch.nn.Linear(state_size, mid_size)
        self.mid2mean = torch.nn.Linear(mid_size, action_size)
        self.mid2logv = torch.nn.Linear(mid_size, action_size)
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
        m = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))
        action = m.rsample()
        log_prob = m.log_prob(action)

        return action, mean, log_prob

    def get_log_prob(self, obs, action):
        mean, std = self.forward(obs)
        m = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))
        log_prob = m.log_prob(action)

        return action, mean, log_prob

class Agent:
    def __init__(self, state_size, n_latent, device):
        self.device = device

        self.policy = Policy_network(state_size, n_latent, 1024)
        self.qf1 = Q_network(n_latent, 1024)
        self.qf2 = Q_network(n_latent, 1024)
        self.target_qf1 = Q_network(n_latent, 1024)
        self.target_qf2 = Q_network(n_latent, 1024)
        self.log_alpha = torch.tensor([0.1], requires_grad=True, device=device)

        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        self.to(device)

        self.target_qf1.eval()
        self.target_qf2.eval()

        self.policy_opt = torch.optim.Adam(self.policy.parameters())
        self.qf1_opt = torch.optim.Adam(self.qf1.parameters())
        self.qf2_opt = torch.optim.Adam(self.qf2.parameters())
        self.alpha_opt = torch.optim.Adam([self.log_alpha])

        self.qf_criterion = torch.nn.MSELoss()

        self.target_entropy = float(-n_latent)
        self.tau = 5e-3

    def to(self, device):
        self.policy.to(device)
        self.qf1.to(device)
        self.qf2.to(device)
        self.target_qf1.to(device)
        self.target_qf2.to(device)

    def train(self):
        self.policy.train()
        self.qf1.train()
        self.qf2.train()
        
    def eval(self):
        self.policy.eval()
        self.qf1.eval()
        self.qf2.eval()

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
        self.target_qf1.eval()
        self.target_qf2.eval()

    def learn_step(self, action, reward, state_size, i, writer=None):
        batch_size = action.shape[0]
        state = torch.randn(batch_size, state_size, device=self.device)

        new_obs_action, _, log_prob = self.policy.sample(state)
        # log_prob = log_prob.unsqueeze(-1)
        q1 = self.qf1(action)
        q2 = self.qf2(action)

        min_q = torch.min(self.qf1(new_obs_action), self.qf2(new_obs_action))
        alpha = torch.exp(self.log_alpha)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        policy_loss = (alpha * log_prob - min_q).mean()
        qf1_loss = self.qf_criterion(q1, reward)
        qf2_loss = self.qf_criterion(q2, reward)

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        self.qf1_opt.zero_grad()
        qf1_loss.backward()
        self.qf1_opt.step()

        self.qf2_opt.zero_grad()
        qf2_loss.backward()
        self.qf2_opt.step()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # target_qfの更新
        for t_p, p in zip(self.target_qf1.parameters(), self.qf1.parameters()):
            t_p.data.copy_(t_p.data * (1.0-self.tau) + p.data * self.tau)
        for t_p, p in zip(self.target_qf2.parameters(), self.qf2.parameters()):
            t_p.data.copy_(t_p.data * (1.0-self.tau) + p.data * self.tau)

        if writer is not None:
            writer.add_scalar("val/log_alpha", self.log_alpha.item(), i)
            writer.add_scalar("loss/alpha", alpha_loss.item(), i)
            writer.add_scalar("loss/policy", policy_loss.item(), i)
            writer.add_scalar("loss/qf1", qf1_loss.item(), i)
            writer.add_scalar("loss/qf2", qf2_loss.item(), i)

    def act(self, state_size):
        state = torch.randn(1, state_size, device=self.device)
        action, _, _ = self.policy.sample(state)
        return action

    def mean_act(self, state_size):
        state = torch.randn(1, state_size, device=self.device)
        _, action, _ = self.policy.sample(state)
        return action


def get_collate_fn():
    def _f(batch):
        action = torch.cat([item["action"] for item in batch], dim=0)
        reward = torch.cat([item["reward"] for item in batch], dim=0)
        return action, reward
    return _f

def get_dataloader(dataset, batchsize):
    return torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            sampler=torch.utils.data.sampler.RandomSampler(dataset),
            collate_fn=get_collate_fn()
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--spm_model", required=True)
    parser.add_argument("--grammar_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_experiment", type=int, default=10)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    writer = SummaryWriter(log_dir=args.output_dir)

    config = Config()
    config.load_json(str(Path(args.vae_checkpoint).with_name("hyper_param.json")))
    config.dropout = 0.0

    device = torch.device("cuda", args.gpu)

    tester = VAE_tester(config, sp, device)
    tester.load_pt(args.vae_checkpoint)

    state_size = config.n_latent
    agent = Agent(state_size, config.n_latent, device)

    with open(args.grammar_data, "rt", encoding="utf-8") as f:
        data = json.load(f)
    memory = [{"action":tester.encode(item["utterance"]).cpu(), "reward":torch.tensor([item["grammar"]])} for item in data]
    # data = []
    # memory = []

    repeatedly = re.compile(r"(.+)\1{3}")
    head = re.compile("^[,ぁァぃィぅゥぇェぉォヵヶゃャゅュょョゎヮ」』ー)]")

    i = 0
    # dataloader = get_dataloader(memory, 64)
    # agent.train()
    # for _ in range(20):
    #     for action, reward in dataloader:
    #         action = action.to(device)
    #         reward = reward.to(device)
    #         agent.learn_step(action, reward, state_size, i, writer)
    #         i += 1

    reward_history = dict()

    for epoch in range(args.num_epoch):
        if len(memory) > 0:
            print("*** #{} learn from memory ***".format(epoch))
            dataloader = get_dataloader(memory[:64*512], 64)
            agent.train()
            for action, reward in dataloader:
                action = action.to(device)
                reward = reward.to(device)
                agent.learn_step(action, reward, state_size, i, writer)
                i += 1

        print("*** experiment ***")
        agent.eval()
        rewards = 0.0
        with torch.no_grad():
            for _ in range(args.num_experiment):
                action = agent.act(state_size)
                utt = tester.beam_generate(action, 5)[0]
                if repeatedly.search(utt) is not None or head.search(utt) is not None or len(utt.strip()) == 0:
                    print("reward from predefined")
                    reward = 0.0
                elif utt in reward_history:
                    print("reward from history")
                    reward = reward_history[utt]
                else:
                    reward = 1.0
                    # print(utt)
                    # while True:
                    #     try:
                    #         reward = float(input())
                    #     except ValueError:
                    #         print("try again")
                    #     else:
                    #         if reward < 0 or reward > 1:
                    #             print("reward must be 0 <= r <= 1")
                    #         else:
                    #             break
                    # reward_history[utt] = reward
                memory.append({"action":action.cpu(), "reward":torch.tensor([reward])})
                data.append({"utterance":utt, "grammar":reward})
                rewards += reward
            writer.add_scalar("experiment/avg_reward", rewards/args.num_experiment, epoch)

        torch.save(agent.state_dict(), str(Path(args.output_dir)/"epoch{:04d}.pt".format(epoch)))

    with open(str(Path(args.output_dir)/"updated_memory.json"), "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
