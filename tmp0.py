# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fire import Fire
from torch import nn
from torchvision import datasets, transforms


class CLI:
    def __init__(
        self,
        ds_path="~/mnist/ds",
        base_model_path="~/mnist/base_model.pt",
        device="cuda",
        bs=32,
        lr=2e-4,
        n_steps=100,
        eval_interval=10,
    ):
        torch.manual_seed(0)
        self._ds_path = Path(ds_path).expanduser()
        self._base_model_path = Path(base_model_path).expanduser()
        self._device = device
        self._bs = bs
        self._lr = lr
        self._n_steps = n_steps
        self._eval_interval = eval_interval

    def convert_mnist(self):
        _convert_mnist(self._ds_path / "train", train=True)
        _convert_mnist(self._ds_path / "val", train=False)

    def pretrain(self):
        train_data = _get_mnist(self._ds_path / "train/a.npz", self._bs, shuffle=True)
        val_data = _get_mnist(self._ds_path / "val/a.npz", self._bs, shuffle=False)

        model = Model()
        model.apply(_init_params)

        self._train(model, train_data, val_data)

        torch.save(model.state_dict(), self._base_model_path)

    def finetune(self):
        train_data = _get_mnist(self._ds_path / "train/b.npz", self._bs, shuffle=True)
        val_data = _get_mnist(self._ds_path / "val/b.npz", self._bs, shuffle=False)

        model = Model()
        model.load_state_dict(torch.load(self._base_model_path, weights_only=True))

        self._train(model, train_data, val_data)

    def _train(self, model, train_data, val_data):
        model = model.to(self._device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self._lr)

        for i, (x, y) in enumerate(train_data):
            x, y = _normalize_img(x.to(self._device)), y.to(self._device)
            opt.zero_grad()
            out = model(x)
            loss = F.nll_loss(out, y.detach())
            loss.backward()
            opt.step()
            print(i, loss.item())

            if i % self._eval_interval == 0:
                eval_loss, accuracy = _evaluate(model, val_data, self._device)
                print("EVAL", i, eval_loss, accuracy)
                model.train()

            if i == self._n_steps:
                break


@torch.no_grad
def _evaluate(model, val_data, device, max_steps=32):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for i, (x, y) in enumerate(val_data):
        if i == max_steps:
            break
        x, y = _normalize_img(x.to(device)), y.to(device)
        out = model(x)
        loss = F.nll_loss(out, y)
        total_loss += loss
        pred = out.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item()
        total += x.shape[0]
    avg_loss = total_loss / min(max_steps, len(val_data))
    return avg_loss.item(), 100.0 * correct / total


class Model(nn.Module):
    def __init__(
        self,
        dim=256,
        adapter=None,
        bias=False,
        rank=2,
        alpha=16,
        dropout=0.0,
        quantize=False,
        decompose=False,
    ):
        super().__init__()
        self._pre = nn.Sequential(nn.Linear(28 * 28, dim), nn.SiLU())
        self._post = nn.Sequential(nn.SiLU(), nn.Linear(dim, 10))

        if adapter is None:
            self._main = nn.Linear(dim, dim, bias=bias)
        elif adapter == "lora":
            self._main = LoRALinear(dim, dim, rank, alpha, dropout, bias, quantize)
        else:
            raise NotImplementedError()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self._pre(x)
        x = self._main(x)
        x = self._post(x)
        return F.log_softmax(x, dim=1)


def _init_params(m):
    if hasattr(m, "initialize_parameters") and callable(m.initialize_parameters):
        m.initialize_parameters()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)


class _MnistDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self._data = data
        self._n = len(self._data["x"])
        assert self._n == len(self._data["y"])

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return self._data["x"][i], self._data["y"][i]


def _get_mnist(path, bs, shuffle):
    data = np.load(path)
    ds = _MnistDataset(data)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )
    return loader


def _normalize_img(x):
    return x.to(torch.float32) / 127.5 - 1


def _convert_mnist(out_dir, train):
    tmp_path = Path("/tmp/mnist_data")
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            tmp_path,
            train=train,
            download=not tmp_path.exists(),
            transform=transforms.ToTensor(),
        ),
        batch_size=None,
        num_workers=0,
        shuffle=False,
    )

    data = {"a": {"x": [], "y": []}, "b": {"x": [], "y": []}}
    for x, y in loader:
        x = (x * 255).clamp(0, 255).to(torch.uint8).numpy()
        if y < 4:
            splits = ["a"]
        elif y < 7:
            splits = ["a", "b"]
        else:
            splits = ["b"]
        for split in splits:
            data[split]["x"].append(x)
            data[split]["y"].append(y)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, ds in data.items():
        for k in ds:
            ds[k] = np.stack(ds[k])
            print(
                train, split, k, ds[k].shape, ds[k].dtype, np.min(ds[k]), np.max(ds[k])
            )
        np.savez(out_dir / f"{split}.npz", **ds)


if __name__ == "__main__":
    Fire(CLI)
