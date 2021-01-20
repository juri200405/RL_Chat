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
from MeCab import Tagger

from vae_check import VAE_tester
from config import Config
from agent import Agent
from dataloader import get_dataloader
from is_in_corpus import MyModel

def manual_reward():
    def _f(state_ids, state_utt, action_ids, action_utt):
        print(action_utt)
        r = -1.0
        while r < 0.0 or r > 1.0:
            try:
                r = float(input("reward (0 <= r <= 1) : "))
            except ValueError:
                print("try again")
        return r
    return _f

def len_reward(target_len, len_range=0, token=True):
    low = target_len - len_range
    high = target_len + len_range
    if token:
        def _f(state_ids, state_utt, action_ids, action_utt):
            if len(action_ids) < low or len(action_ids) > high:
                return 0.0
            else:
                return 1.0
    else:
        def _f(state_ids, state_utt, action_ids, action_utt):
            if len(action_utt) < low or len(action_utt) > high:
                return 0.0
            else:
                return 1.0
    return _f

def repeat_reward(repeat_num=3):
    repeatedly = re.compile(r"(.+)\1{{{}}}".format(repeat_num-1))
    def _f(state_ids, state_utt, action_ids, action_utt):
        if repeatedly.search(action_utt) is not None:
            return 0.0
        else:
            return 1.0
    return _f

def norm_bleu_reward():
    m = Tagger()
    def norm_list(utt):
        return [word for word, w_type in [tuple(item.split('\t')) for item in m.parse(utt).strip().split('\n') if item!="EOS"] if w_type.split(',')[0]=="名詞"]

    def _f(state_ids, state_utt, action_ids, action_utt):
        state_norm = norm_list(state_utt)
        action_norm = norm_list(action_utt)
        return bleu_score.sentence_bleu([state_norm], action_norm, smoothing_function=bleu_score.SmoothingFunction().method1, weights=(0.5, 0.5))
    return _f

def norm_rate_reward():
    m = Tagger()
    def norm_set(utt):
        return {word for word, w_type in [tuple(item.split('\t')) for item in m.parse(utt).strip().split('\n') if item!="EOS"] if w_type.split(',')[0]=="名詞"}

    def _f(state_ids, state_utt, action_ids, action_utt):
        state_norm = norm_set(state_utt)
        action_norm = norm_set(action_utt)
        union = state_norm | action_norm
        intersection = state_norm & action_norm
        return len(intersection)/len(union) if len(union)>0 else 1.0
    return _f

def token_bleu_score(sp):
    def _f(state_ids, state_utt, action_ids, action_utt):
        return bleu_score.sentence_bleu(
                [[sp.id_to_piece(item) for item in state_ids]],
                [sp.id_to_piece(item) for item in action_ids],
                smoothing_function=bleu_score.SmoothingFunction().method1,
                weights=(0.5, 0.5)
                )
    return _f

def char_bleu_score():
    def _f(state_ids, state_utt, action_ids, action_utt):
        return bleu_score.sentence_bleu(
                [list(state_utt)],
                list(action_utt),
                smoothing_function=bleu_score.SmoothingFunction().method1,
                weights=(0.5, 0.5)
                )
    return _f

def input_len_reward(len_range=0, token=True):
    if token:
        def _f(state_ids, state_utt, action_ids, action_utt):
            low = len(state_ids) - len_range
            high = len(state_ids) + len_range
            if len(action_ids) < low or len(action_ids) > high:
                return 0.0
            else:
                return 1.0
    else:
        def _f(state_ids, state_utt, action_ids, action_utt):
            low = len(state_utt) - len_range
            high = len(state_utt) + len_range
            if len(action_utt) < low or len(action_utt) > high:
                return 0.0
            else:
                return 1.0
    return _f

class Environment():
    def __init__(
            self,
            tester,
            reward_funcs
            ):
        self.tester = tester
        self.reward_funcs = reward_funcs

    def calc_reward(self, state, action, state_ids):
        state_utt = self.tester.sp.decode(state_ids)
        action_ids = self.tester.beam_generate_ids(action, 5)[0]
        action_utt = self.tester.sp.decode(action_ids)

        data_dict = {"utterance": action_utt, "epoch": epoch, "step": step, "input": state_utt}
        if state_utt == action_utt:
            reward = 0.0
        else:
            reward = 0.0
            for r_type, func, weight in self.reward_funcs:
                f_reward = func(state_ids, state_utt, action_ids, action_utt) * weight
                data_dict[r_type] = f_reward
                reward += f_reward

        data_dict["reward"] = reward
        return reward, data_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--spm_model", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--mid_size", type=int, default=1024)
    parser.add_argument("--num_experiment", type=int, default=10)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--training_num", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--initial_log_alpha", type=float, default=1e-4)
    parser.add_argument(
            "--reward_type",
            nargs='+',
            choices=[
                "manual",
                "char_len_reward",
                "token_len_reward",
                "repeat_reward",
                "norm_bleu_reward",
                "norm_rate_reward",
                "token_bleu_reward",
                "char_bleu_reward",
                "input_char_len_reward",
                "input_token_len_reward",
                ],
            )
    parser.add_argument("--reward_weight", type=int, nargs='*', default=[])
    parser.add_argument("--target_len", type=int, default=10)
    parser.add_argument("--target_range", type=int, default=0)
    parser.add_argument("--repeat_num", type=int, default=3)
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
    corpus_datas = [item for item in corpus_datas if len(item)>0]

    agent = Agent(
            torch.nn.Sigmoid(),
            config.n_latent,
            args.mid_size,
            device,
            lr=args.lr,
            discount=args.discount,
            initial_log_alpha=args.initial_log_alpha,
            target_entropy=-config.n_latent
            )

    with open(str(Path(args.output_dir)/"arguments.json"), "wt", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if len(args.reward_type) != len(args.reward_weight) and len(args.reward_weight) != 0:
        print("incorrect num reward_weight")
        exit()

    if len(args.reward_weight) == 0:
        reward_weight = [1.0/len(args.reward_type)] * len(args.reward_type)
    else:
        reward_weight = args.reward_weight

    reward_funcs = []
    for r_type, r_weight in zip(args.reward_type, reward_weight):
        if r_type == "manual":
            func = manual_reward()
        elif r_type == "char_len_reward":
            func = len_reward(args.target_len, len_range=args.target_range, token=False)
        elif r_type == "token_len_reward":
            func = len_reward(args.target_len, len_range=args.target_range, token=True)
        elif r_type == "repeat_reward":
            func = repeat_reward(repeat_num=args.repeat_num)
        elif r_type == "norm_bleu_reward":
            func = norm_bleu_reward()
        elif r_type == "norm_rate_reward":
            func = norm_rate_reward()
        elif r_type == "token_bleu_reward":
            func = token_bleu_reward(sp)
        elif r_type == "char_bleu_reward":
            func = char_bleu_reward()
        elif r_type == "input_char_len_reward":
            func = input_len_reward(len_range=args.target_range, token=False)
        elif r_type == "input_token_len_reward":
            func = input_len_reward(len_range=args.target_range, token=True)

        reward_funcs.append((r_type, func, r_weight))

    env = Environment(
            tester,
            reward_funcs
            )

    data = []
    memory = []

    i = 0
    for epoch in range(args.num_epoch):
        if len(memory) > 0:
            print("*** #{} learn from memory ***".format(epoch))
            sample = random.sample(memory, min(128*args.training_num, len(memory)))
            dataloader = get_dataloader(sample, 128, use_hidden=False)
            agent.train()
            for batch in dataloader:
                graph = True if i ==0 else False
                state, action, reward, next_state, is_final = batch
                result_dict, losses = agent.learn(state, action, reward, next_state, is_final, graph=graph)

                if losses is not None:
                    input_list = [
                            ("state", state),
                            ("action", action),
                            ("reward", reward),
                            ("next_state", next_state),
                            ("is_final", is_final)
                            ]
                    param_list = list(agent.policy.named_parameters()) \
                            + list(agent.qf1.named_parameters()) \
                            + list(agent.qf2.named_parameters()) \
                            + list(agent.target_qf1.named_parameters()) \
                            + list(agent.target_qf2.named_parameters()) \
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
            for step in range(args.num_experiment):
                if "state" in memory_dict:
                    memory_dict["next_state"] = state.detach().cpu()
                    memory_dict["is_final"] = torch.tensor([0.0])
                    memory.append(memory_dict)
                    memory_dict = dict()

                action = agent.act(state)
                reward, data_dict = env.calc_reward(state, action, state_ids)

                memory_dict["state"] = state.detach().cpu()
                memory_dict["action"] = action.detach().cpu()
                memory_dict["reward"] = torch.tensor([reward])

                state_ids = random.choice(corpus_datas)
                state = tester.encode_from_ids(state_ids)

                data.append(data_dict)
                rewards += reward

            memory_dict["next_state"] = state.cpu()
            memory_dict["is_final"] = torch.tensor([1.0])
            memory.append(memory_dict)

            writer.add_scalar("experiment/average_reward", rewards/args.num_experiment, epoch)

        with open(str(Path(args.output_dir)/"history_{:05d}.json".format(epoch)), "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        data.clear()
        if epoch % 1000 == 0:
            torch.save(agent.state_dict(), str(Path(args.output_dir)/"epoch{:02d}k.pt".format(epoch//1000)))
    torch.save(agent.state_dict(), str(Path(args.output_dir)/"final.pt"))
