import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn
from torch.utils.data.sampler import RandomSampler

def get_collate_fn(use_hidden):

    if use_hidden:
        def _f(batch):
            state = torch.cat([item["state"] for item in batch], dim=0)
            hidden = torch.cat([item["hidden"] for item in batch], dim=0)
            action = torch.cat([item["action"] for item in batch], dim=0)
            reward = torch.cat([item["reward"] for item in batch], dim=0)
            next_state = torch.cat([item["next_state"] for item in batch], dim=0)
            next_hidden = torch.cat([item["next_hidden"] for item in batch], dim=0)
            is_final = torch.cat([item["is_final"] for item in batch], dim=0)

            return state, hidden, action, reward, next_state, next_hidden, is_final
    else:
        def _f(batch):
            state = torch.cat([item["state"] for item in batch], dim=0)
            action = torch.cat([item["action"] for item in batch], dim=0)
            reward = torch.cat([item["reward"] for item in batch], dim=0)
            next_state = torch.cat([item["next_state"] for item in batch], dim=0)
            is_final = torch.cat([item["is_final"] for item in batch], dim=0)

            return state, action, reward, next_state, is_final

    return _f


def get_dataloader(dataset, batchsize, use_hidden=True):
    dataloader = data.DataLoader(
            dataset,
            batch_size = batchsize,
            sampler=RandomSampler(dataset),
            collate_fn=get_collate_fn(use_hidden)
            )
    return dataloader
