import random
import json

class Database():
    def __init__(self, init_file=None):
        self.memory = []
        self.init_memory = []
        if init_file is not None:
            for item in init_file:
                with open(item, "rt", encoding="utf-8") as f:
                    self.init_memory += json.load(f)

    def __len__(self):
        return len(self.memory) + len(self.init_memory)

    def sample(self, num):
        memory = self.memory + self.init_memory
        if num > len(memory):
            batch = random.sample(memory, len(memory))
            batch += random.choices(memory, k=num-len(memory))
        else:
            batch = random.sample(memory, num)

        return batch

    def push(self, item):
        self.memory.append(item)

    def save_added_memory(self, output_file):
        with open(output_file, 'wt', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)
