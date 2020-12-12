import unicodedata

import torch
import transformers

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def normalize_string(s):
    '''
    文s中に含まれる文字を正規化。
    '''
    s = ''.join(c for c in unicodedata.normalize('NFKC', s) if unicodedata.category(c) != 'Mn')
    return s

class ChatSystem:
    def __init__(
            self,
            database,
            agent,
            sp,
            encoder,
            decoder,
            dbdc,
            encoder_device,
            decoder_device,
            dbdc_device,
            num_beams=5,
            sample_size=256,
            max_len=128
            ):
        # keyboard = [InlineKeyboardButton(" 0 ", callback_data='0'),
        #         InlineKeyboardButton(" 1 ", callback_data='1'),
        #         InlineKeyboardButton(" 2 ", callback_data='2')]
        # self.reply_markup = InlineKeyboardMarkup([keyboard])

        self.states = dict()
        self.database = database
        self.agent = agent
        self.dialog_count = 0

        self.sample_size = sample_size
        self.num_beams = num_beams
        self.max_len = max_len

        self.sp = sp
        self.encoder = encoder.to(encoder_device)
        self.decoder = decoder.to(decoder_device)
        self.dbdc = dbdc.to(dbdc_device)
        self.encoder.eval()
        self.decoder.eval()
        self.dbdc.eval()

        self.encoder_device = encoder_device
        self.decoder_device = decoder_device
        self.dbdc_device = dbdc_device

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
                state = self.encode(input_dict["utt"])
                action, hidden = self.agent.act(state, session_state["hidden"])
                score = self.calc_score(session_state["memory"], state, action)
                session_state["memory"].append({
                    "state":state,
                    "hidden":session_state["hidden"],
                    "action":action,
                    "score":score
                    })
                session_state["hidden"] = hidden

                utt = "sys: " + self.decode(action) + " : score : " + str(score)
                # output_dict = {'utt':utt, 'markup': self.reply_markup}
                output_dict = {'utt':utt, 'markup': None}
            else:
                output_dict = {'utt': '[err] 対話を開始してください', 'markup': None}
        else:
            output_dict = {'utt': '[err] 対話を開始してください', 'markup': None}

        return output_dict

    def button(self, reward):
        pass

    def calc_score(self, memory, state, action):
        latent_list = []
        for item in memory:
            latent_list.append(item["state"])
            latent_list.append(item["action"])
        latent_list.append(state)
        latent_list.append(action)

        conv_len = torch.tensor([len(latent_list)], device=self.dbdc_device)
        x = torch.cat(latent_list, dim=1).to(self.dbdc_device)
        return self.dbdc(x, conv_len).item()

    def encode(self, utt):
        utt = normalize_string(utt)
        with torch.no_grad():
            input_s = torch.tensor([1] + self.sp.encode(utt) + [2], dtype=torch.long, device=self.encoder_device).unsqueeze(0)
            inp_mask = torch.tensor([[False]*input_s.shape[1] + [True]*(self.max_len - input_s.shape[1])], device=self.encoder_device)
            pad = torch.full((1, self.max_len - input_s.shape[1]), 3, dtype=torch.long, device=self.encoder_device)
            input_s = torch.cat((input_s, pad), dim=1)
            state = self.encoder(input_s, attention_mask=inp_mask) # state = (batch(1), n_latent)
        return state.unsqueeze(0).cpu()

    def decode(self, action):
        if self.num_beams < 2:
            sys_utt = self.greedy_decode(action)
        else:
            sys_utt = self.beam_decode(action)
        return sys_utt[0]

    def greedy_decode(self, memory):
        with torch.no_grad():
            tgt = torch.full((memory.shape[0], 1), 1, dtype=torch.long, device=self.decoder_device)  # <s>
            unfinish = torch.ones(memory.shape[0], 1, dtype=torch.long, device=self.decoder_device)
            memory = memory.to(self.decoder_device)
            while tgt.shape[1] < self.max_len:
                out = self.decoder(tgt, memory)
                _, topi = out.transpose(0,1).topk(1)
                next_word = topi[:,-1]
                next_word = next_word*unfinish + (3)*(1-unfinish)
                tgt = torch.cat((tgt, next_word), dim=-1)
                unfinish = unfinish * (~(next_word == 2)).long()
                if unfinish.max() == 0: # </s>
                    break
        return self.sp.decode(tgt.cpu().tolist())

    def beam_decode(self, memory):
        num_beams = self.num_beams
        batch_size = memory.shape[0]
        vocab_size = len(self.sp)
        beam_scorer = transformers.BeamSearchScorer(batch_size=batch_size, max_length=self.max_len, num_beams=num_beams, device=self.decoder_device)
        with torch.no_grad():
            memory_list = torch.split(memory, 1)
            memory_list = [torch.cat([item]*num_beams) for item in memory_list]
            memory = torch.cat(memory_list)
            memory = memory.to(self.decoder_device)

            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=self.decoder_device)
            beam_scores[:,1:] = -1e9
            beam_scores = beam_scores.view((batch_size * num_beams,))

            tgt = torch.full((memory.shape[0], 1), 1, dtype=torch.long, device=self.decoder_device)  # <s>
            while tgt.shape[1] < self.max_len:
                out = self.decoder(tgt, memory) # (seq, batch, n_vocab)
                out = out.transpose(0,1)[:, -1, :]

                next_token_scores = torch.nn.functional.log_softmax(out, dim=-1)
                next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

                next_token_scores, next_tokens = torch.topk(next_token_scores, 2*num_beams, dim=1, largest=True, sorted=True)
                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size

                beam_outputs = beam_scorer.process(tgt, next_token_scores, next_tokens, next_indices, pad_token_id=3, eos_token_id=2)
                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                tgt = torch.cat((tgt[beam_idx, :], beam_next_tokens.unsqueeze(-1)), dim=-1)

                if beam_scorer.is_done:
                    break
            decoded = beam_scorer.finalize(tgt, beam_scores, next_tokens, next_indices, pad_token_id=3, eos_token_id=2)
        return self.sp.decode(decoded.cpu().tolist())
