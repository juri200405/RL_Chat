import argparse
from threading import Thread
import json
import re

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import sentencepiece as spm

import encoder_decoder
from agent import Chat_Module
from database import Database
from chat_system import ChatSystem

from my_telegram_bot import TelegramBot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--setting_file", required=True)
    parser.add_argument("-m", "--spm_model", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    # ハイパーパラメータ
    parser.add_argument("-p", "--hyper_param", required=True)

    # エンコーダデコーダのパラメータ
    parser.add_argument("--pt_file")
    
    # BERTのpre-trainモデルへのパス
    parser.add_argument("--bert_path")

    parser.add_argument("--database")

    parser.add_argument("--telegram", action="store_true")
    
    args = parser.parse_args()

    with open(args.hyper_param, 'rt') as f:
        hyperp = json.load(f)

    encoder = encoder_decoder.Bert_Encoder_gru(args.bert_path)
    decoder = encoder_decoder.transformer_Decoder(
            hyperp["n_vocab"],
            hyperp["d_model"],
            hyperp["n_head"],
            hyperp["n_hidden"],
            hyperp["decoder_nlayers"],
            encoder.bert.get_input_embeddings(),
            nn.LayerNorm(hyperp["d_model"]),
            dropout=hyperp["dropout"]
            )

    if args.pt_file is not None:
        checkpoint = torch.load(args.pt_file)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    writer = SummaryWriter(args.output_dir)

    database = Database(args.database)
    agent = Chat_Module(
            sp,
            encoder,
            decoder,
            encoder_device=torch.device("cuda", 0),
            decoder_device=torch.device("cuda", 0),
            learning_agent_device=torch.device("cuda", 1),
            chat_agent_device=torch.device("cuda", 1),
            batch_size=hyperp["batch_size"],
            writer=writer
            )

    Thread(target=agent.learn, daemon=True).start()

    system = ChatSystem(database, agent, hyperp["sample_size"])
    
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
