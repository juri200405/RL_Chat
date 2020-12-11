import argparse
from threading import Thread
import json
import re
from pathlib import Path

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import sentencepiece as spm

import encoder_decoder
from agent import Chat_Module
from database import Database
from chat_system import ChatSystem
from config import Config

from my_telegram_bot import TelegramBot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--setting_file", required=True)
    parser.add_argument("-m", "--spm_model", required=True)
    parser.add_argument("-o", "--output_dir", required=True)

    # エンコーダデコーダのパラメータ
    parser.add_argument("--vae_checkpoint")
    
    parser.add_argument("--database", nargs='*')

    parser.add_argument("--telegram", action="store_true")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sample_size", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=5)

    parser.add_argument("--obs_size", type=int, default=1024)
    
    args = parser.parse_args()

    hyper_param = Path(args.vae_checkpoint).with_name("hyper_param.json")
    config = Config()
    config.load_json(hyper_param)
    config.dropout = 0.0

    embedding = encoder_decoder.Transformer_Embedding(config)
    encoder = encoder_decoder.transformer_Encoder(config, embedding, nn.LayerNorm(config.d_model))
    decoder = encoder_decoder.transformer_Decoder(config, embedding, nn.LayerNorm(config.d_model))

    if args.vae_checkpoint is not None:
        checkpoint = torch.load(args.vae_checkpoint, map_location="cpu")
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.eval()
    decoder.eval()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    writer = SummaryWriter(args.output_dir)

    database = Database(args.database)
    agent = Chat_Module(
            sp,
            encoder,
            decoder,
            encoder_device=torch.device("cuda", 0),
            decoder_device=torch.device("cuda", 0),
            learning_agent_device=torch.device("cuda", 0),
            chat_agent_device=torch.device("cuda", 0),
            batch_size=args.batch_size,
            n_latent=config.n_latent,
            max_len=config.max_len,
            obs_size=args.obs_size,
            num_beams=args.num_beams,
            writer=writer
            )

    Thread(target=agent.learn, daemon=True).start()

    system = ChatSystem(
            database,
            agent,
            args.sample_size,
            output_file=str(Path(args.output_dir)/"chat_database.json")
            )
    
    if args.telegram:
        bot = TelegramBot(system, args.setting_file)
        bot.run()
    else:
        while True:
            utt = input(">")
            m = re.match(r"/(?P<command>.+)", utt)
            if m is not None:
                if m.group("command") == "start":
                    print(system.initial_message({"utt":"", "sessionId":0})["utt"])
                elif m.group("command") == "end":
                    print(system.end_message({"utt":"", "sessionId":0})["utt"])
                elif m.group("command") == "reset":
                    print(system.reset_message({"utt":"", "sessionId":0})["utt"])
                elif m.group("command") == "quit":
                    break
                else:
                    print("command error")
            else:
                print(system.reply({"utt":utt, "sessionId":0})["utt"])
