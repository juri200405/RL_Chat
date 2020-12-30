import argparse
from pathlib import Path
import json
import re
import random

import torch
from torch.utils.tensorboard import SummaryWriter

import sentencepiece as spm
from nltk import bleu_score
from torchviz import make_dot

from vae_check import VAE_tester
from config import Config
from agent import Agent
from dataloader import get_dataloader
from is_in_corpus import MyModel

class Environment():
    def __init__(self, tester, additional_reward, init_data=None, manual_reward=False, reward_model=None):
        self.history_list = []
        self.tester = tester

        self.repeatedly = re.compile(r"(.+)\1{3}")
        self.head = re.compile(r"^[,ぁァぃィぅゥぇェぉォヵヶゃャゅュょョゎヮ」』ー)]")
        self.left = re.compile(r"[「『(]")
        self.right = re.compile(r"[」』)]")

        self.additional_reward = additional_reward
        if additional_reward in ["cos", "state_action_cos"]:
            self.cos = torch.nn.CosineSimilarity(dim=1)

        self.manual_reward = manual_reward
        self.reward_model = reward_model

        self.use_memory = init_data is not None
        self.init_data = init_data

    def reset(self):
        self.history_list.clear()

    def _reward(self, utt):
        if len(utt.strip()) == 0:
            return 0.0
        if self.repeatedly.search(utt) is not None:
            return 0.0
        if self.head.match(utt) is not None:
            return 0.0
        if len(self.left.findall(utt)) != len(self.right.findall(utt)):
            return 0.0

        if self.manual_reward:
            print(utt)
            r = -1.0
            while r < 0 or r > 1:
                try:
                    r = float(input("reward (0 <= r <= 1) : "))
                except ValueError:
                    print("try again")
            return r
        else:
            return 1.0

    def calc_reward(self, state, action):
        utt = self.tester.beam_generate(action, 5)[0]
        if self.reward_model is not None:
            pre = self.reward_model(action).item()
        else:
            pre = self._reward(utt)
        data_dict = {"utterance": utt, "pre": pre, "epoch": epoch, "step": step}
        t_action = None
        t_reward = None

        if self.use_memory:
            t_utt, t_action, t_pre = random.choice(self.init_data)

        if self.additional_reward == "none":
            reward = pre
            if self.use_memory:
                t_reward = t_pre

        elif self.additional_reward == "bleu":
            if len(utt) == 0:
                bleu = 1
            elif len(self.history_list) > 0:
                bleu = bleu_score.sentence_bleu(self.utt_list, list(utt), smoothing_function=bleu_score.SmoothingFunction().method1, weights=(0.5, 0.5))
            else:
                bleu = 0
            data_dict["bleu"] = bleu
            reward = pre - bleu

            if self.use_memory:
                if len(t_utt) == 0:
                    t_bleu = 1
                elif len(self.history_list) > 0:
                    t_bleu = bleu_score.sentence_bleu(self.utt_list, list(t_utt), smoothing_function=bleu_score.SmoothingFunction().method1, weights=(0.5, 0.5))
                else:
                    t_bleu = 0
                t_reward = t_pre - t_bleu

            if len(utt) > 0:
                self.history_list.append(list(utt))

        elif self.additional_reward == "cos":
            if len(self.history_list) > 0:
                x1 = torch.cat(self.history_list, dim=0)
                x2 = action.expand_as(x1)
                cs = self.cos(x1, x2).mean().item() / 2 + 0.5
                if self.use_memory:
                    xt = t_action.expand_as(x1)
                    t_cs = self.cos(x1, xt).mean().item() / 2 + 0.5
            else:
                cs = 0
                if self.use_memory:
                    t_cs = 0
            data_dict["cos"] = cs
            reward = pre - cs
            if self.use_memory:
                t_reward = t_pre - t_cs
            self.history_list.append(action.detach())

        elif self.additional_reward == "state_action_cos":
            cs = self.cos(state, action).item() / 2 + 0.5
            reward = (1.3 * pre + 0.7 * cs) - 1.0
            data_dict["cos"] = cs
            if self.use_memory:
                t_cs = self.cos(state, t_action).item() / 2 + 0.5
                t_reward = (1.3 * t_pre + 0.7 * t_cs) - 1.0

        data_dict["reward"] = reward
        return data_dict, t_action, t_reward


def sqrt_activation(x):
    sign = torch.sign(x)
    abs_x = torch.abs(x)
    root = torch.sqrt(abs_x)
    return torch.where(abs_x < 1, x, sign*root)

def none_activation(x):
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--reward_checkpoint", required=True)
    parser.add_argument("--spm_model", required=True)
    parser.add_argument("--grammar_data", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mid_size", type=int, default=1024)
    parser.add_argument("--num_experiment", type=int, default=10)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--training_num", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--initial_log_alpha", type=float, default=1e-4)
    parser.add_argument("--no_gru", action='store_true')
    parser.add_argument("--use_history_hidden", action='store_true')
    parser.add_argument("--random_state", action='store_true')
    parser.add_argument("--manual_reward", action='store_true')
    parser.add_argument("--activation", choices=["sqrt", "sigmoid", "none", "tanh"], default="none")
    parser.add_argument("--additional_reward", choices=["none", "bleu", "cos", "state_action_cos"], default="none")
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.output_dir)
    print("complete making writer")

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    print("complete loading spm_model")

    config = Config()
    config.load_json(str(Path(args.vae_checkpoint).with_name("hyper_param.json")))
    config.dropout = 0.0
    print("complete loading config")

    device = torch.device("cuda", args.gpu)

    tester = VAE_tester(config, sp, device)
    tester.load_pt(args.vae_checkpoint)
    print("complete loading vae")

    if args.activation == "none":
        activation_function = none_activation
    elif args.activation == "sqrt":
        activation_function = sqrt_activation
    elif args.activation == "sigmoid":
        activation_function = torch.nn.Sigmoid()
    elif args.activation == "tanh":
        activation_function = torch.nn.Tanh()

    obs_size = 64
    agent = Agent(
            activation_function,
            config.n_latent,
            obs_size,
            args.mid_size,
            device,
            lr=args.lr,
            discount=args.discount,
            initial_log_alpha=args.initial_log_alpha,
            no_gru=args.no_gru,
            target_entropy=-config.n_latent
            )

    for i in range(100):
        writer.add_scalar("debug/activation", activation_function(torch.tensor([float(i-50)/10.0])).item(), i)

    with open(str(Path(args.output_dir)/"arguments.json"), "wt", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if args.grammar_data is not None:
        use_memory = True
        with open(args.grammar_data, "rt", encoding="utf-8") as f:
            init_data = json.load(f)
        init_data = [(item["utterance"], tester.encode(item["utterance"]), item["grammar"]) for item in init_data]
    else:
        use_memory = False
        init_data = None

    reward_model = MyModel(config.n_latent, 2048).to(device)
    reward_model.eval()
    reward_model.load_state_dict(torch.load(args.reward_checkpoint, map_location=device))
    env = Environment(tester, args.additional_reward, init_data, args.manual_reward, reward_model)

    data = []
    memory = []

    i = 0
    for epoch in range(args.num_epoch):
        if len(memory) > 0:
            print("*** #{} learn from memory ***".format(epoch))
            sample = random.sample(memory, min(64*args.training_num, len(memory)))
            dataloader = get_dataloader(sample, 64, use_hidden=args.use_history_hidden)
            agent.train()
            hidden = torch.zeros(64, obs_size, device=device)
            for batch in dataloader:
                graph = True if i ==0 else False
                if args.use_history_hidden:
                    state, hidden, action, reward, next_state, next_hidden, is_final = batch
                    _, result_dict, losses = agent.learn(*batch, graph=graph, use_history_hidden=True)
                else:
                    state, action, reward, next_state, is_final = batch
                    hidden, result_dict, losses = agent.learn(state, hidden, action, reward, next_state, None, is_final, graph=graph, use_history_hidden=False)

                if losses is not None:
                    input_list = [
                            ("state", state),
                            # ("hidden", hidden),
                            ("action", action),
                            ("reward", reward),
                            ("next_state", next_state),
                            # ("next_hidden", next_hidden),
                            ("is_final", is_final)
                            ]
                    param_list = list(agent.policy.named_parameters()) \
                            + list(agent.qf1.named_parameters()) \
                            + list(agent.qf2.named_parameters()) \
                            + list(agent.target_qf1.named_parameters()) \
                            + list(agent.target_qf2.named_parameters()) \
                            + list(agent.gru.named_parameters()) \
                            + [("log_alpha", agent.log_alpha)]

                    for name, item in losses.items():
                        make_dot(item, params=dict(input_list+param_list)).render(str(Path(args.output_dir)/name))
                        # make_dot(item, params=dict(param_list)).render(str(Path(args.output_dir)/name))

                for name, item in result_dict.items():
                    writer.add_scalar(name, item, i)
                i += 1

        print("*** experiment ***")
        agent.eval()
        rewards = 0.0
        memory_dict = dict()
        if use_memory:
            t_memory_dict = dict()

        with torch.no_grad():
            state = torch.randn(1, config.n_latent, device=device).tanh()
            hidden = torch.zeros(1, obs_size, device=device)
            for step in range(args.num_experiment):
                # エージェントの出力行動
                if "state" in memory_dict:
                    memory_dict["next_state"] = state.detach().cpu()
                    memory_dict["next_hidden"] = hidden.detach().cpu()
                    memory_dict["is_final"] = torch.tensor([0.0])
                    memory.append(memory_dict)
                    memory_dict = dict()

                if use_memory:
                    # メモリからの行動
                    if "state" in t_memory_dict:
                        t_memory_dict["next_state"] = t_state.detach().cpu()
                        t_memory_dict["next_hidden"] = hidden.detach().cpu()
                        t_memory_dict["is_final"] = torch.tensor([0.0])
                        memory.append(t_memory_dict)
                        t_memory_dict = dict()

                action, next_hidden = agent.act(state, hidden)
                data_dict, t_action, t_reward = env.calc_reward(state, action)

                # エージェントの出力行動
                memory_dict["state"] = state.detach().cpu()
                memory_dict["hidden"] = hidden.detach().cpu()
                memory_dict["action"] = action.detach().cpu()
                memory_dict["reward"] = torch.tensor([data_dict["reward"]])

                if use_memory:
                    # メモリからの行動
                    t_memory_dict["state"] = state.detach().cpu()
                    t_memory_dict["hidden"] = hidden.detach().cpu()
                    t_memory_dict["action"] = t_action.detach().cpu()
                    t_memory_dict["reward"] = torch.tensor([t_reward])

                if args.random_state:
                    state = torch.randn_like(state).tanh()
                    if use_memory:
                        t_state = torch.randn_like(state).tanh()
                else:
                    state = action.detach()
                    if use_memory:
                        t_state = t_action.detach()

                hidden = next_hidden.detach()
                data.append(data_dict)
                rewards += data_dict["reward"]

            # エージェントの出力行動
            memory_dict["next_state"] = state.cpu()
            memory_dict["next_hidden"] = hidden.cpu()
            memory_dict["is_final"] = torch.tensor([1.0])
            memory.append(memory_dict)

            if use_memory:
                # メモリからの行動
                t_memory_dict["next_state"] = t_state.cpu()
                t_memory_dict["next_hidden"] = hidden.cpu()
                t_memory_dict["is_final"] = torch.tensor([1.0])
                memory.append(t_memory_dict)

            writer.add_scalar("experiment/total_reward", rewards, epoch)
            env.reset()

        torch.save(agent.state_dict(), str(Path(args.output_dir)/"epoch{:04d}.pt".format(epoch)))

    with open(str(Path(args.output_dir)/"updated_memory.json"), "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
