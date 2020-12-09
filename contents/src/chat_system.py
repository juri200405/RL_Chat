import unicodedata

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def normalize_string(s):
    '''
    文s中に含まれる文字を正規化。
    '''
    s = ''.join(c for c in unicodedata.normalize('NFKC', s) if unicodedata.category(c) != 'Mn')
    return s

class ChatSystem:
    def __init__(self, database, agent, sample_size=256, output_file="chat_database.json"):
        # keyboard = [InlineKeyboardButton(" 0 ", callback_data='0'),
        #         InlineKeyboardButton(" 1 ", callback_data='1'),
        #         InlineKeyboardButton(" 2 ", callback_data='2')]
        # self.reply_markup = InlineKeyboardMarkup([keyboard])

        self.states = dict()
        self.database = database
        self.agent = agent
        self.dialog_count = 0

        self.sample_size = sample_size

        self.database_output_file = output_file

    def initial_message(self, input_dict):
        self.states[input_dict["sessionId"]] = {"inprogress":True, "hidden":self.agent.initial_hidden(), "memory":[]}
        self.dialog_count += 1
        if self.dialog_count > 2:
            self.agent.update_param(self.database.sample(self.sample_size))
            self.dialog_count = 0
        output_dict = {'utt': '[start] 対話を開始しました', 'markup': None}
        return output_dict

    def end_message(self, input_dict):
        session_state = self.states[input_dict["sessionId"]]

        self.database.push(session_state["memory"])
        self.database.save_added_memory(self.database_output_file)
        session_state["inprogress"] = False

        output_dict = {'utt': '[end] 対話を終了しました', 'markup': None}
        return output_dict

    def reset_message(self, input_dict):
        self.end_message(input_dict)
        self.initial_message(input_dict)
        output_dict = {'utt': '[reset] 対話をリセットしました', 'markup': None}
        return output_dict

    def reply(self, input_dict):
        if input_dict["sessionId"] in self.states:
            session_state = self.states[input_dict["sessionId"]]
            if session_state["inprogress"]:
                usr_utt = normalize_string(input_dict["utt"])
                utt, hidden = self.agent.make_utt(usr_utt, session_state["hidden"])
                session_state["hidden"] = hidden
                session_state["memory"].append({"usr_utt":usr_utt, "sys_utt":utt})
                # output_dict = {'utt':utt, 'markup': self.reply_markup}
                utt = "sys: " + utt
                output_dict = {'utt':utt, 'markup': None}
            else:
                output_dict = {'utt': '[err] 対話を開始してください', 'markup': None}
        else:
            output_dict = {'utt': '[err] 対話を開始してください', 'markup': None}

        return output_dict

    def button(self, reward):
        pass
