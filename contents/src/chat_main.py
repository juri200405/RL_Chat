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
from dbdc_model import DBDC

from my_telegram_bot import TelegramBot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--setting_file", required=True)
    parser.add_argument("-o", "--output_dir", required=True)

    parser.add_argument("--dbdc_checkpoint", required=True)
    
    parser.add_argument("--database", nargs='*')

    parser.add_argument("--telegram", action="store_true")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sample_size", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=5)

    parser.add_argument("--obs_size", type=int, default=1024)
    
    args = parser.parse_args()

    with open(str(Path(args.dbdc_checkpoint).with_name("dbdc_model_param.json")), "rt", encoding="utf-8") as f:
        dbdc_param = json.load(f)
    spm_model = dbdc_param["spm_model"]
    vae_checkpoint = dbdc_param["vae_checkpoint"]

    hyper_param = Path(vae_checkpoint).with_name("hyper_param.json")
    config = Config()
    config.load_json(hyper_param)
    config.dropout = 0.0

    embedding = encoder_decoder.Transformer_Embedding(config)
    encoder = encoder_decoder.transformer_Encoder(config, embedding, nn.LayerNorm(config.d_model))
    decoder = encoder_decoder.transformer_Decoder(config, embedding, nn.LayerNorm(config.d_model))

    checkpoint = torch.load(vae_checkpoint, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    dbdc = DBDC(config.n_latent, dbdc_param["gru_hidden"], dbdc_param["ff_hidden"])
    checkpoint = torch.load(args.dbdc_checkpoint, map_location="cpu")
    dbdc.load_state_dict(checkpoint["model_state_dict"])

    sp = spm.SentencePieceProcessor(model_file=spm_model)

    writer = SummaryWriter(args.output_dir)

    # database = Database(args.database)
    database = Database()
    agent = Chat_Module(
            learning_agent_device=torch.device("cuda", 3),
            chat_agent_device=torch.device("cuda", 3),
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
            sp,
            encoder,
            decoder,
            dbdc,
            encoder_device=torch.device("cuda", 2),
            decoder_device=torch.device("cuda", 2),
            dbdc_device=torch.device("cuda", 2),
            num_beams=5,
            sample_size=args.sample_size,
            max_len=config.max_len
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
