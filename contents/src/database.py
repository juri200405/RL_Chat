import random
import json

class Database():
    def __init__(self, init_file=None):
        if init_file is None:
            self.memory = []
        else:
            with open(init_file, "rt", encoding="utf-8") as f:
                self.memory = json.load(f)

    def __len__(self):
        return len(self.memory)

    def sample(self, num):
        if num > len(self.memory):
            batch = random.sample(self.memory, len(self.memory))
            batch += random.choices(self.memory, k=num-len(self.memory))
        else:
            batch = random.sample(self.memory, num)

        return batch

    def push(self, item):
        self.memory.append(item)
