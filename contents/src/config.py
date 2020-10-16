import json

import torch

class Config:
    def __init__(self):
        self._n_vocab = 16000
        self._d_model = 256
        self._n_hidden = 2048
        self._encoder_nlayers = 2
        self._decoder_nlayers = 2
        self._n_head = 2
        self._max_len = 128

        self._anneal_k = 0.00005
        self._x0_epoch = 20

        self._batch_size = 64
        self._num_epoch = 200

        self._model_type = "transformer"
        self._optim_type = "Adam"

        self._dropout = 0.1

        self._encoder_device = torch.device('cpu')
        self._decoder_device = torch.device('cpu')
        self._device_name = 'cpu'

        self._model_list = ["bert", "transformer"]
        self._optim_list = ["Adam", "RAdam", "Yogi"]

    @property
    def n_vocab(self) -> int:
        return self._n_vocab
    @n_vocab.setter
    def n_vocab(self, value: int) -> None:
        self._n_vocab = value

    @property
    def d_model(self) -> int:
        return self._d_model
    @d_model.setter
    def d_model(self, value: int) -> None:
        self._d_model = value

    @property
    def n_hidden(self) -> int:
        return self._n_hidden
    @n_hidden.setter
    def n_hidden(self, value: int) -> None:
        self._n_hidden = value

    @property
    def encoder_nlayers(self) -> int:
        return self._encoder_nlayers
    @encoder_nlayers.setter
    def encoder_nlayers(self, value: int) -> None:
        self._encoder_nlayers = value

    @property
    def decoder_nlayers(self) -> int:
        return self._decoder_nlayers
    @decoder_nlayers.setter
    def decoder_nlayers(self, value: int) -> None:
        self._decoder_nlayers = value

    @property
    def n_head(self) -> int:
        return self._n_head
    @n_head.setter
    def n_head(self, value: int) -> None:
        self._n_head = value

    @property
    def max_len(self) -> int:
        return self._max_len
    @max_len.setter
    def max_len(self, value: int) -> None:
        self._max_len = value

    @property
    def anneal_k(self) -> float:
        return self._anneal_k
    @anneal_k.setter
    def anneal_k(self, value: float) -> None:
        self._anneal_k = value

    @property
    def x0_epoch(self) -> int:
        return self._x0_epoch
    @x0_epoch.setter
    def x0_epoch(self, value: int) -> None:
        self._x0_epoch = value

    @property
    def batch_size(self) -> int:
        return self._batch_size
    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self._batch_size = value

    @property
    def num_epoch(self) -> int:
        return self._num_epoch
    @num_epoch.setter
    def num_epoch(self, value: int) -> None:
        self._num_epoch = value

    @property
    def model_type(self) -> str:
        return self._model_type
    @model_type.setter
    def model_type(self, value: str) -> None:
        if value in self._model_list:
            self._model_type = value

    @property
    def optim_type(self) -> str:
        return self._optim_type
    @optim_type.setter
    def optim_type(self, value: str) -> None:
        if value in self._optim_list:
            self._optim_type = value

    @property
    def dropout(self) -> float:
        return self._dropout
    @dropout.setter
    def dropout(self, value: float) -> None:
        self._dropout = value

    @property
    def encoder_device(self) -> torch.device:
        return self._encoder_device
    @encoder_device.setter
    def encoder_device(self, value: torch.device) -> None:
        self._encoder_device = value

    @property
    def decoder_device(self) -> torch.device:
        return self._decoder_device
    @decoder_device.setter
    def decoder_device(self, value: torch.device) -> None:
        self._decoder_device = value


    def load_json(self, filename: str) -> None:
        with open(filename, 'rt', encoding='utf-8') as f:
            hyperp = json.load(f)

        self.n_vocab = hyperp["n_vocab"]
        self.d_model = hyperp["d_model"]
        self.n_hidden = hyperp["n_hidden"]
        self.encoder_nlayers = hyperp["encoder_nlayers"]
        self.decoder_nlayers = hyperp["decoder_nlayers"]
        self.n_head = hyperp["n_head"]
        self.dropout = hyperp["dropout"]
        self.model_type = hyperp["model_type"]
        self.batch_size = hyperp["batch_size"]
        self.optim_type = hyperp["optim_type"]
        self.max_len = hyperp["max_len"]
        self.anneal_k = hyperp["anneal_k"]
        self.num_epoch = hyperp["num_epoch"]
        self.x0_epoch = hyperp["x0_epoch"]

        use_gpus = hyperp["use_gpus"]

        device_num = torch.cuda.device_count()
        if len(use_gpus) > 1:
            if device_num > 1:
                self.encoder_device = torch.device('cuda', use_gpus[0])
                self.decoder_device = torch.device('cuda', use_gpus[1])
                self._device_name = "cuda"
            elif device_num == 1:
                self.encoder_device = self.decoder_device = torch.device('cuda', use_gpus[0])
                self._device_name = "cuda"
            else:
                self.encoder_device = self.decoder_device = torch.device('cpu')
                self._device_name = "cpu"
        elif len(use_gpus) == 1:
            if device_num >= 1:
                self.encoder_device = self.decoder_device = torch.device('cuda', use_gpus[0])
                self._device_name = "cuda"
            else:
                self.encoder_device = self.decoder_device = torch.device('cpu')
                self._device_name = "cpu"
        else:
            self.encoder_device = self.decoder_device = torch.device('cpu')
            self._device_name = "cpu"

    def save_json(self, filename: str) -> None:
        hyperp = dict()

        hyperp["n_vocab"] = self.n_vocab
        hyperp["d_model"] = self.d_model
        hyperp["n_hidden"] = self.n_hidden
        hyperp["encoder_nlayers"] = self.encoder_nlayers
        hyperp["decoder_nlayers"] = self.decoder_nlayers
        hyperp["n_head"] = self.n_head
        hyperp["dropout"] = self.dropout
        hyperp["model_type"] = self.model_type
        hyperp["batch_size"] = self.batch_size
        hyperp["optim_type"] = self.optim_type
        hyperp["max_len"] = self.max_len
        hyperp["anneal_k"] = self.anneal_k
        hyperp["num_epoch"] = self.num_epoch
        hyperp["x0_epoch"] = self.x0_epoch

        if self._device_name == "cpu":
            hyperp["use_gpus"] = []
        else:
            e = self.encoder_device.index
            d = self.decoder_device.index
            if e == d:
                hyperp["use_gpus"] = [e]
            else:
                hyperp["use_gpus"] = [e, d]

        with open(filename, 'wt', encoding='utf-8') as f:
            json.dump(hyperp, f)
