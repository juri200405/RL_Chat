import argparse
from pathlib import Path
import json
import re
import random
import pickle

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
    def __init__(
            self,
            tester,
            reward_type,
            additional_reward,
            additional_reward_rate,
            reward_model=None,
            corpus_ngram=None,
            datas=None,
            target_len=None,
            low=None,
            high=None
            ):
        self.tester = tester

        self.reward_type = reward_type
        if reward_type == "reward_model":
            self.reward_model = reward_model
        elif reward_type in ["corpus_ngram", "weighted_corpus_ngram"]:
            self.corpus_ngram = corpus_ngram
            self.max_ngrams = max(corpus_ngram.values())
        elif reward_type == "corpus_bleu":
            self.datas = datas
        elif reward_type in ["char_len_reward", "token_len_reward"]:
            self.target_len = target_len
        elif reward_type in ["char_len_range_reward", "token_len_range_reward"]:
            self.low = low
            self.high = high
        elif reward_type == "repeat_reward":
            self.repeatedly = re.compile(r"(.+)\1{2}")
        elif reward_type == "sentence_head_reward":
            self.head = re.compile(r"私は")

        self.additional_reward = additional_reward
        if additional_reward in ["pos_state_action_cos", "neg_state_action_cos"]:
            self.cos = torch.nn.CosineSimilarity(dim=1)
        self.additional_reward_rate = additional_reward_rate

    def manual_reward(self, utt):
        print(utt)
        r = -1.0
        while r < 0.0 or r > 1.0:
            try:
                r = float(input("reward (0 <= r <= 1) : "))
            except ValueError:
                print("try again")
        return r

    def corpus_ngram_reward(self, ids, use_weight):
        n_gram = [tuple(ids[i:i+3]) for i in range(len(ids)-2)]
        score = 0.0
        for item in n_gram:
            if item in self.corpus_ngram:
                if use_weight:
                    score += (self.corpus_ngram[item] / self.max_ngrams)
                else:
                    score += 1.0
        return (score / len(n_gram)) if len(n_gram) > 0 else 0.0

    def corpus_bleu(self, text):
        return bleu_score.sentence_bleu([list(item) for item in random.sample(self.datas, 100)], list(text), smoothing_function=bleu_score.SmoothingFunction().method1)

    def len_reward(self, text):
        if len(text) == self.target_len:
            return 1.0
        else:
            return 0.0

    def len_range_reward(self, text):
        if len(text) < self.low or len(text) > self.high:
            return 0.0
        else:
            return 1.0

    def repeat_reward(self, utt):
        if self.repeatedly.search(utt) is not None:
            return 0.0
        else:
            return 1.0

    def sentence_head_reward(self, utt):
        if self.head.match(utt) is not None:
            return 1.0
        else:
            return 0.0
    def calc_reward(self, state, action, state_ids):
        input_utt = self.tester.sp.decode(state_ids)
        ids = self.tester.beam_generate_ids(action, 5)[0]
        utt = self.tester.sp.decode(ids)

        if input_utt == utt:
            pre = 0
        elif self.reward_type == "manual":
            pre = self.manual_reward(utt)
        elif self.reward_type == "reward_model":
            pre = self.reward_model(action).item()
        elif self.reward_type == "corpus_ngram":
            pre = self.corpus_ngram_reward(ids, False)
        elif self.reward_type == "weighted_corpus_ngram":
            pre = self.corpus_ngram_reward(ids, True)
        elif self.reward_type == "corpus_bleu":
            pre = self.corpus_bleu(utt)
        elif self.reward_type == "char_len_reward":
            pre = self.len_reward(utt)
        elif self.reward_type == "token_len_reward":
            pre = self.len_reward(ids)
        elif self.reward_type == "char_len_range_reward":
            pre = self.len_range_reward(utt)
        elif self.reward_type == "token_len_range_reward":
            pre = self.len_range_reward(ids)
        elif self.reward_type == "repeat_reward":
            pre = self.repeat_reward(utt)
        elif self.reward_type == "sentence_head_reward":
            pre = self.sentence_head_reward(utt)
        data_dict = {"utterance": utt, "pre": pre, "epoch": epoch, "step": step, "input": input_utt}

        if self.additional_reward == "none":
            reward = pre
        elif self.additional_reward == "state_action_id_bleu":
            if len(utt) == 0:
                bleu = 1
            else:
                bleu = bleu_score.sentence_bleu([state_ids], ids, smoothing_function=bleu_score.SmoothingFunction().method1, weights=(0.5, 0.5))
            data_dict["bleu"] = bleu
            reward = pre - bleu
        elif self.additional_reward == "state_action_char_bleu":
            if len(utt) == 0:
                bleu = 1
            else:
                bleu = bleu_score.sentence_bleu([list(input_utt)], list(utt), smoothing_function=bleu_score.SmoothingFunction().method1, weights=(0.5, 0.5))
            data_dict["bleu"] = bleu
            reward = pre - bleu
        elif self.additional_reward == "pos_state_action_cos":
            cs = self.cos(state, action).item() / 2 + 0.5
            data_dict["cos"] = cs
            reward = (1-self.additional_reward_rate)*pre + self.additional_reward_rate*cs
        elif self.additional_reward == "neg_state_action_cos":
            cs = self.cos(state, action).item() / 2 + 0.5
            data_dict["cos"] = cs
            reward = pre - cs

        data_dict["reward"] = reward
        return data_dict


def sqrt_activation(x):
    sign = torch.sign(x)
    abs_x = torch.abs(x)
    root = torch.sqrt(abs_x)
    return torch.where(abs_x < 1, x, sign*root)

def none_activation(x):
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--spm_model", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--reward_checkpoint", default=None)
    parser.add_argument("--ngram_file", default=None)
    parser.add_argument("--datas_file", default=None)
    parser.add_argument("--mid_size", type=int, default=1024)
    parser.add_argument("--num_experiment", type=int, default=10)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--training_num", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--initial_log_alpha", type=float, default=1e-4)
    parser.add_argument("--activation", choices=["sqrt", "sigmoid", "none", "tanh"], default="none")
    parser.add_argument(
            "--reward_type",
            choices=[
                "manual",
                "reward_model",
                "corpus_ngram",
                "weighted_corpus_ngram",
                "corpus_bleu",
                "char_len_reward",
                "token_len_reward",
                "repeat_reward",
                "sentence_head_reward",
                "char_len_range_reward",
                "token_len_range_reward"
                ],
            default="corpus_ngram"
            )
    parser.add_argument(
            "--additional_reward",
            choices=["none", "state_action_id_bleu", "state_action_char_bleu", "pos_state_action_cos", "neg_state_action_cos"],
            default="none"
            )
    parser.add_argument("--additional_reward_rate", type=float, default=0.5)
    parser.add_argument("--target_len", type=int, default=-1)
    parser.add_argument("--low", type=int, default=-1)
    parser.add_argument("--high", type=int, default=-1)
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

    with open(args.input_file, "rb") as f:
        corpus_datas = pickle.load(f)

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
            no_gru=True,
            target_entropy=-config.n_latent
            )

    with open(str(Path(args.output_dir)/"arguments.json"), "wt", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if (args.reward_checkpoint is not None) and (args.reward_type == "reward_model"):
        reward_model = MyModel(config.n_latent, 2048).to(device)
        reward_model.eval()
        reward_model.load_state_dict(torch.load(args.reward_checkpoint, map_location=device))
    else:
        reward_model = None

    if (args.ngram_file is not None) and (args.reward_type in ["corpus_ngram", "weighted_corpus_ngram"]):
        with open(args.ngram_file, "rb") as f:
            corpus_ngram, total_ngrams = pickle.load(f)
    else:
        corpus_ngram = None

    if (args.datas_file is not None) and (args.reward_type == "corpus_bleu"):
        with open(args.datas_file, "rt", encoding="utf-8") as f:
            datas = [item.strip() for item in f if len(item.strip())>0]
    else:
        datas = None

    if (args.target_len > 0) and (args.reward_type in ["char_len_reward", "token_len_reward"]):
        target_len = args.target_len
    else:
        target_len = None

    if (args.low > 0) and (args.high > 0) and (args.reward_type in ["char_len_range_reward", "token_len_range_reward"]):
        low = args.low
        high = args.high
    else:
        low = None
        high = None

    env = Environment(
            tester,
            args.reward_type,
            args.additional_reward,
            args.additional_reward_rate,
            reward_model=reward_model,
            corpus_ngram=corpus_ngram,
            datas=datas,
            target_len=target_len,
            low=low,
            high=high
            )

    data = []
    memory = []

    i = 0
    for epoch in range(args.num_epoch):
        if len(memory) > 0:
            print("*** #{} learn from memory ***".format(epoch))
            sample = random.sample(memory, min(64*args.training_num, len(memory)))
            dataloader = get_dataloader(sample, 64, use_hidden=False)
            agent.train()
            hidden = torch.zeros(64, obs_size, device=device)
            for batch in dataloader:
                graph = True if i ==0 else False
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

        with torch.no_grad():
            state_ids = random.choice(corpus_datas)
            state = tester.encode_from_ids(state_ids)
            hidden = torch.zeros(1, obs_size, device=device)
            for step in range(args.num_experiment):
                if "state" in memory_dict:
                    memory_dict["next_state"] = state.detach().cpu()
                    memory_dict["next_hidden"] = hidden.detach().cpu()
                    memory_dict["is_final"] = torch.tensor([0.0])
                    memory.append(memory_dict)
                    memory_dict = dict()

                action, next_hidden = agent.act(state, hidden)
                data_dict = env.calc_reward(state, action, state_ids)

                memory_dict["state"] = state.detach().cpu()
                memory_dict["hidden"] = hidden.detach().cpu()
                memory_dict["action"] = action.detach().cpu()
                memory_dict["reward"] = torch.tensor([data_dict["reward"]])

                state_ids = random.choice(corpus_datas)
                state = tester.encode_from_ids(state_ids)

                hidden = next_hidden.detach()
                data.append(data_dict)
                rewards += data_dict["reward"]

            memory_dict["next_state"] = state.cpu()
            memory_dict["next_hidden"] = hidden.cpu()
            memory_dict["is_final"] = torch.tensor([1.0])
            memory.append(memory_dict)

            writer.add_scalar("experiment/total_reward", rewards, epoch)

        torch.save(agent.state_dict(), str(Path(args.output_dir)/"epoch{:05d}.pt".format(epoch)))
        with open(str(Path(args.output_dir)/"history_{:05d}.json".format(epoch)), "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        data.clear()
