import argparse
from pathlib import Path
import json
import re
import random

import torch
from torch.utils.tensorboard import SummaryWriter

import sentencepiece as spm
from nltk import bleu_score

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

        return 2.0

    return _function


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

    obs_size = 128
    agent = Agent(config.n_latent, obs_size, device, lr=1e-4)

    data = []
    memory = []

    is_predefined = get_grammra_reward_function()

    i = 0
    for epoch in range(args.num_epoch):
        if len(memory) > 0:
            print("*** #{} learn from memory ***".format(epoch))
            sample = random.sample(memory, min(64*32, len(memory)))
            dataloader = get_dataloader(sample, 64)
            agent.train()
            for batch in dataloader:
                result_dict = agent.learn(*batch)
                for name, item in result_dict.items():
                    writer.add_scalar(name, item, i)
                i += 1

        print("*** experiment ***")
        agent.eval()
        rewards = 0.0
        memory_dict = dict()
        utt_list = []
        with torch.no_grad():
            state = torch.randn(1, 1, config.n_latent, device=device)
            hidden = torch.zeros(1, 1, obs_size, device=device)
            for _ in range(args.num_experiment):
                if "state" in memory_dict:
                    memory_dict["next_state"] = state.detach().cpu()
                    memory_dict["next_hidden"] = hidden.detach().cpu()
                    memory_dict["is_final"] = torch.tensor([0.0])
                    memory.append(memory_dict)
                    memory_dict = dict()

                action, next_hidden = agent.act(state, hidden)
                utt = tester.beam_generate(action, 5)[0]

                pre = is_predefined(utt)
                if len(utt) == 0:
                    bleu = 1
                elif len(utt_list) > 0:
                    bleu = bleu_score.sentence_bleu(utt_list, list(utt), smoothing_function=bleu_score.SmoothingFunction().method1)
                else:
                    bleu = 0

                if len(utt) > 0:
                    utt_list.append(list(utt))
                reward = pre + (1-bleu)

                memory_dict["state"] = state.detach().cpu()
                memory_dict["hidden"] = hidden.detach().cpu()
                memory_dict["action"] = action.detach().cpu()
                memory_dict["reward"] = torch.tensor([reward])
                state = action.unsqueeze(0).detach()
                hidden = next_hidden.detach()
                data.append({"utterance":utt, "reward":reward, "bleu":bleu, "pre": pre})
                rewards += reward

            memory_dict["next_state"] = state.cpu()
            memory_dict["next_hidden"] = hidden.cpu()
            memory_dict["is_final"] = torch.tensor([1.0])
            memory.append(memory_dict)
            writer.add_scalar("experiment/total_reward", rewards, epoch)

        torch.save(agent.state_dict(), str(Path(args.output_dir)/"epoch{:04d}.pt".format(epoch)))

    with open(str(Path(args.output_dir)/"updated_memory.json"), "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
