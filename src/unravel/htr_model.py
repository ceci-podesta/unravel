"""Modelo HTR (CNN + BiLSTM + CTC).

Adaptado de github.com/georgeretsi/HTR-best-practices (DAS 2022).
Arquitectura idéntica al original para garantizar compatibilidad con los
pesos preentrenados (`htrnet.pt`).
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CNN(nn.Module):
    def __init__(self, cnn_cfg: list, flattening: str = "maxpool") -> None:
        super().__init__()
        self.k = 1
        self.flattening = flattening
        self.features = nn.ModuleList([nn.Conv2d(1, 32, 7, [4, 2], 3), nn.ReLU()])
        in_channels = 32
        cntm = 0
        cnt = 1
        for m in cnn_cfg:
            if m == "M":
                self.features.add_module(f"mxp{cntm}", nn.MaxPool2d(2, 2))
                cntm += 1
            else:
                for _ in range(int(m[0])):
                    x = int(m[1])
                    self.features.add_module(f"cnv{cnt}", BasicBlock(in_channels, x))
                    in_channels = x
                    cnt += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for nn_module in self.features:
            y = nn_module(y)
        if self.flattening == "maxpool":
            y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k // 2])
        elif self.flattening == "concat":
            y = y.view(y.size(0), -1, 1, y.size(3))
        return y


class CTCtopR(nn.Module):
    def __init__(self, input_size: int, rnn_cfg: tuple, nclasses: int, rnn_type: str = "gru") -> None:
        super().__init__()
        hidden, num_layers = rnn_cfg
        if rnn_type == "lstm":
            self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=0.2)
        elif rnn_type == "gru":
            self.rec = nn.GRU(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=0.2)
        else:
            raise ValueError(f"unknown rnn_type: {rnn_type}")
        self.fnl = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * hidden, nclasses))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(2, 3, 0, 1)[0]
        y = self.rec(y)[0]
        return self.fnl(y)


class CTCtopB(nn.Module):
    def __init__(self, input_size: int, rnn_cfg: tuple, nclasses: int, rnn_type: str = "gru") -> None:
        super().__init__()
        hidden, num_layers = rnn_cfg
        if rnn_type == "lstm":
            self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=0.2)
        elif rnn_type == "gru":
            self.rec = nn.GRU(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=0.2)
        else:
            raise ValueError(f"unknown rnn_type: {rnn_type}")
        self.fnl = nn.Sequential(nn.Dropout(0.5), nn.Linear(2 * hidden, nclasses))
        self.cnn = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(input_size, nclasses, (1, 3), 1, (0, 1)),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x.permute(2, 3, 0, 1)[0]
        y = self.rec(y)[0]
        y = self.fnl(y)
        return y, self.cnn(x).permute(2, 3, 0, 1)[0]


class HTRNet(nn.Module):
    def __init__(self, arch_cfg: SimpleNamespace, nclasses: int) -> None:
        super().__init__()
        if arch_cfg.stn:
            raise NotImplementedError("STN no implementado")
        self.stn = None
        cnn_cfg = arch_cfg.cnn_cfg
        self.features = CNN(arch_cfg.cnn_cfg, flattening=arch_cfg.flattening)
        if arch_cfg.flattening in {"maxpool", "avgpool"}:
            hidden = cnn_cfg[-1][-1]
        elif arch_cfg.flattening == "concat":
            hidden = 2 * 8 * cnn_cfg[-1][-1]
        else:
            raise ValueError(f"unknown flattening: {arch_cfg.flattening}")
        head = arch_cfg.head_type
        if head == "rnn":
            self.top = CTCtopR(hidden, (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers), nclasses, rnn_type=arch_cfg.rnn_type)
        elif head == "both":
            self.top = CTCtopB(hidden, (arch_cfg.rnn_hidden_size, arch_cfg.rnn_layers), nclasses, rnn_type=arch_cfg.rnn_type)
        else:
            raise ValueError(f"unknown head_type: {head}")

    def forward(self, x: torch.Tensor):
        if self.stn is not None:
            x = self.stn(x)
        y = self.features(x)
        return self.top(y)


def default_arch_cfg() -> SimpleNamespace:
    """Configuración del modelo según `config.yaml` del repo original."""
    return SimpleNamespace(
        cnn_cfg=[[2, 64], "M", [3, 128], "M", [2, 256]],
        head_type="both",
        rnn_type="lstm",
        rnn_layers=3,
        rnn_hidden_size=256,
        flattening="maxpool",
        stn=False,
    )
