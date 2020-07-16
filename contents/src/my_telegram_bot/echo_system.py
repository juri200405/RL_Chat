import argparse

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from telegram_bot import TelegramBot

class EchoSystem:
    def __init__(self):
        keyboard = [InlineKeyboardButton("Option 1", callback_data='1'),
                InlineKeyboardButton("Option 2", callback_data='2')]
        self.reply_markup = InlineKeyboardMarkup([keyboard])

    def initial_message(self, input_dir):
        output_dir = {'utt': '入力内容をそのまま返すボットです。', 'markup': None}
        return output_dir

    def end_message(self, input_dir):
        output_dir = {'utt': 'ありがとーねー', 'markup': None}
        return output_dir

    def reply(self, input_dir):
        output_dir = {'utt':input_dir['utt'], 'markup': self.reply_markup}
        return output_dir

    def button(self, query):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--setting_file", required=True)
    args = parser.parse_args()

    system = EchoSystem()
    
    bot = TelegramBot(system, args.setting_file)
    bot.run()
