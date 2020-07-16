import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn
from torch.utils.data.sampler import RandomSampler

def get_collate_fn(pad_index=0, bos_index=1, eos_index=2, fix_max_len=None):

    def _f(batch):
        if fix_max_len is not None:
            tensors = [torch.LongTensor([bos_index] + item[:(fix_max_len-2)] + [eos_index]) for item in batch]
        else:
            tensors = [torch.LongTensor([bos_index] + item + [eos_index]) for item in batch]

        padded_sequence = rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_index)

        ids_len = [item.shape[0] for item in tensors]
        inp_padded_mask = [[1.0]*i + [0.0]*(padded_sequence.shape[1]-i) for i in ids_len]
        tgt_padded_mask = [[False]*i + [True]*(padded_sequence.shape[1]-i) for i in ids_len]

        return padded_sequence, torch.tensor(inp_padded_mask), torch.tensor(tgt_padded_mask)

    return _f


def get_dataloader(dataset, batchsize, pad_index=0, bos_index=1, eos_index=2, fix_max_len=None, shuffle=True):
    if shuffle:
        dataloader = data.DataLoader(
                dataset,
                batch_size = batchsize,
                sampler=RandomSampler(dataset),
                collate_fn=get_collate_fn(pad_index, bos_index, eos_index, fix_max_len)
                )
    else:
        dataloader = data.DataLoader(
                dataset,
                batch_size = batchsize,
                shuffle=False,
                collate_fn=get_collate_fn(pad_index, bos_index, eos_index, fix_max_len)
                )
    return dataloader
