import json
from threading import Thread

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler

# class TelegramBot(Thread):
class TelegramBot():
    def __init__(self, system, json_path):
        # super(TelegramBot, self).__init__()
        self.system = system

        with open(json_path, 'rt') as f:
            setting = json.load(f)
        self.TOKEN = setting["TOKEN"]

    def start(self, update, context):
        input_dir = {'utt':None, 'sessionId':str(update.message.from_user.id)}
        system_output = self.system.initial_message(input_dir)
        update.message.reply_text(system_output['utt'], reply_markup=system_output['markup'])

    def end(self, update, context):
        input_dir = {'utt':None, 'sessionId':str(update.message.from_user.id)}
        system_output = self.system.end_message(input_dir)
        update.message.reply_text(system_output['utt'], reply_markup=system_output['markup'])

    def reset(self, update, context):
        input_dir = {'utt':None, 'sessionId':str(update.message.from_user.id)}
        system_output = self.system.reset_message(input_dir)
        update.message.reply_text(system_output['utt'], reply_markup=system_output['markup'])

    def message(self, update, context):
        input_dir = {'utt':update.message.text, 'sessionId':str(update.message.from_user.id)}
        system_output = self.system.reply(input_dir)
        update.message.reply_text(system_output['utt'], reply_markup=system_output['markup'])

    def button(self, update, context):
        query = update.callback_query
        query.answer()
        reward = query.data
        self.system.button(reward)
        query.edit_message_text(text="{}: {}".format(query.message.text, reward))

    def run(self):
        self.updater = Updater(self.TOKEN, use_context=True)
        print(self.updater.bot.get_me())
        self.updater.dispatcher.add_handler(CommandHandler('start', self.start))
        self.updater.dispatcher.add_handler(CommandHandler('end', self.end))
        self.updater.dispatcher.add_handler(CommandHandler('reset', self.reset))
        self.updater.dispatcher.add_handler(CallbackQueryHandler(self.button))
        self.updater.dispatcher.add_handler(MessageHandler(Filters.text, self.message))
        self.updater.start_polling()
        self.updater.idle()
