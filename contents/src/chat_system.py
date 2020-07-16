from telegram import InlineKeyboardButton, InlineKeyboardMarkup

class ChatSystem:
    def __init__(self, database, agent, sample_size=256):
        # keyboard = [InlineKeyboardButton(" 0 ", callback_data='0'),
        #         InlineKeyboardButton(" 1 ", callback_data='1'),
        #         InlineKeyboardButton(" 2 ", callback_data='2')]
        # self.reply_markup = InlineKeyboardMarkup([keyboard])

        self.states = dict()
        self.database = database
        self.agent = agent
        self.dialog_count = 0

        self.sample_size = sample_size

    def initial_message(self, input_dir):
        self.states[input_dir["sessionId"]] = {"inprogress":True, "hidden":self.agent.initial_hidden(), "memory":[]}
        self.dialog_count += 1
        if self.dialog_count > 2:
            self.agent.update_param(self.database.sample(self.sample_size))
            self.dialog_count = 0
        output_dir = {'utt': '[start] 対話を開始しました', 'markup': None}
        return output_dir

    def end_message(self, input_dir):
        session_state = self.states[input_dir["sessionId"]]

        self.database.push(session_state["memory"])
        session_state["inprogress"] = False

        output_dir = {'utt': '[end] 対話を終了しました', 'markup': None}
        return output_dir

    def reset_message(self, input_dir):
        self.end_message(input_dir)
        self.initial_message(input_dir)
        output_dir = {'utt': '[reset] 対話をリセットしました', 'markup': None}
        return output_dir

    def reply(self, input_dir):
        if input_dir["sessionId"] in self.states:
            session_state = self.states[input_dir["sessionId"]]
            if session_state["inprogress"]:
                utt, hidden = self.agent.make_utt(input_dir["utt"], session_state["hidden"])
                session_state["hidden"] = hidden
                session_state["memory"].append({"usr_utt":input_dir["utt"], "sys_utt":utt})
                # output_dir = {'utt':utt, 'markup': self.reply_markup}
                utt = "sys: " + utt
                output_dir = {'utt':utt, 'markup': None}
            else:
                output_dir = {'utt': '[err] 対話を開始してください', 'markup': None}
        else:
            output_dir = {'utt': '[err] 対話を開始してください', 'markup': None}

        return output_dir

    def button(self, reward):
        pass
