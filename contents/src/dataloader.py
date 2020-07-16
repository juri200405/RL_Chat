import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn
from torch.utils.data.sampler import RandomSampler

def get_collate_fn():

    def _f(batch):
        state = torch.cat([item["state"] for item in batch], dim=1)
        hidden = torch.cat([item["hidden"] for item in batch], dim=1)
        action = torch.cat([item["action"] for item in batch], dim=1)
        reward = torch.cat([item["reward"] for item in batch], dim=0)
        next_state = torch.cat([item["next_state"] for item in batch], dim=1)
        next_hidden = torch.cat([item["next_hidden"] for item in batch], dim=1)
        is_final = torch.cat([item["is_final"] for item in batch], dim=0)

        return state, hidden, action, reward, next_state, next_hidden, is_final

    return _f


def get_dataloader(dataset, batchsize):
    dataloader = data.DataLoader(
            dataset,
            batch_size = batchsize,
            sampler=RandomSampler(dataset),
            collate_fn=get_collate_fn()
            )
    return dataloader
