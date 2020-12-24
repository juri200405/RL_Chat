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

def get_grammra_reward_function():
    repeatedly = re.compile(r"(.+)\1{3}")
    head = re.compile(r"^[,ぁァぃィぅゥぇェぉォヵヶゃャゅュょョゎヮ」』ー)]")
    left = re.compile(r"[「『(]")
    right = re.compile(r"[」』)]")

    def _function(utt):
        if len(utt.strip()) == 0:
            return 0.0
        if repeatedly.search(utt) is not None:
            return 0.0
        if head.match(utt) is not None:
            return 0.0
        if len(left.findall(utt)) != len(right.findall(utt)):
            return 0.0

        return 1.0

    return _function


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
    parser.add_argument("--spm_model", required=True)
    parser.add_argument("--grammar_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_experiment", type=int, default=10)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--initial_log_alpha", type=float, default=1e-4)
    parser.add_argument("--no_gru", type=bool, default=False)
    parser.add_argument("--activation", choices=["sqrt", "sigmoid", "none", "tanh"], default="none")
    parser.add_argument("--training_num", type=int, default=32)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    print("complete loading spm_model")

    writer = SummaryWriter(log_dir=args.output_dir)
    print("complete making writer")

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

    with open(args.grammar_data, "rt", encoding="utf-8") as f:
        init_data = json.load(f)
    init_data = [(item["utterance"], tester.encode(item["utterance"]), item["grammar"]) for item in init_data]
    data = []
    memory = []

    is_predefined = get_grammra_reward_function()

    i = 0
    for epoch in range(args.num_epoch):
        if len(memory) > 0:
            print("*** #{} learn from memory ***".format(epoch))
            sample = random.sample(memory, min(64*args.training_num, len(memory)))
            dataloader = get_dataloader(sample, 64)
            agent.train()
            for batch in dataloader:
                graph = True if i ==0 else False
                result_dict, losses = agent.learn(*batch, graph=graph)
                if losses is not None:
                    state, hidden, action, reward, next_state, next_hidden, is_final = batch
                    input_list = [
                            ("state",state),
                            ("hidden",hidden),
                            ("action",action),
                            ("reward",reward),
                            ("next_state",next_state),
                            ("next_hidden",next_hidden),
                            ("is_final",is_final)
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

                for name, item in result_dict.items():
                    writer.add_scalar(name, item, i)
                i += 1

        print("*** experiment ***")
        agent.eval()
        rewards = 0.0
        memory_dict = dict()
        t_memory_dict = dict()
        utt_list = []
        with torch.no_grad():
            state = torch.randn(1, config.n_latent, device=device)
            hidden = torch.zeros(1, obs_size, device=device)
            for _ in range(args.num_experiment):
                # エージェントの出力行動
                if "state" in memory_dict:
                    memory_dict["next_state"] = state.detach().cpu()
                    memory_dict["next_hidden"] = hidden.detach().cpu()
                    memory_dict["is_final"] = torch.tensor([0.0])
                    memory.append(memory_dict)
                    memory_dict = dict()

                # メモリからの行動
                if "state" in t_memory_dict:
                    t_memory_dict["next_state"] = t_state.detach().cpu()
                    t_memory_dict["next_hidden"] = hidden.detach().cpu()
                    t_memory_dict["is_final"] = torch.tensor([0.0])
                    memory.append(t_memory_dict)
                    t_memory_dict = dict()

                action, next_hidden = agent.act(state, hidden)
                utt = tester.beam_generate(action, 5)[0]
                pre = is_predefined(utt)

                t_utt, t_action, t_pre = random.choice(init_data)

                if len(utt) == 0:
                    bleu = 1
                elif len(t_utt) == 0:
                    t_bleu = 1
                elif len(utt_list) > 0:
                    bleu = bleu_score.sentence_bleu(utt_list, list(utt), smoothing_function=bleu_score.SmoothingFunction().method1, weights=(0.5, 0.5))
                    t_bleu = bleu_score.sentence_bleu(utt_list, list(t_utt), smoothing_function=bleu_score.SmoothingFunction().method1, weights=(0.5, 0.5))
                else:
                    bleu = 0
                    t_bleu = 0

                if len(utt) > 0:
                    utt_list.append(list(utt))
                reward = pre - bleu
                t_reward = t_pre - t_bleu

                # エージェントの出力行動
                memory_dict["state"] = state.detach().cpu()
                memory_dict["hidden"] = hidden.detach().cpu()
                memory_dict["action"] = action.detach().cpu()
                memory_dict["reward"] = torch.tensor([reward])
                state = action.detach()

                # メモリからの行動
                t_memory_dict["state"] = state.detach().cpu()
                t_memory_dict["hidden"] = hidden.detach().cpu()
                t_memory_dict["action"] = t_action.detach().cpu()
                t_memory_dict["reward"] = torch.tensor([t_reward])
                t_state = t_action.detach()

                hidden = next_hidden.detach()
                data.append({"utterance":utt, "reward":reward, "bleu":bleu, "pre": pre})
                rewards += reward

            # エージェントの出力行動
            memory_dict["next_state"] = state.cpu()
            memory_dict["next_hidden"] = hidden.cpu()
            memory_dict["is_final"] = torch.tensor([1.0])
            memory.append(memory_dict)

            # メモリからの行動
            t_memory_dict["next_state"] = t_state.cpu()
            t_memory_dict["next_hidden"] = hidden.cpu()
            t_memory_dict["is_final"] = torch.tensor([1.0])
            memory.append(t_memory_dict)
            writer.add_scalar("experiment/total_reward", rewards, epoch)

        torch.save(agent.state_dict(), str(Path(args.output_dir)/"epoch{:04d}.pt".format(epoch)))

    with open(str(Path(args.output_dir)/"updated_memory.json"), "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
