"""按序列分组的乱序采样器。

默认的 shuffle=True 会让相邻 batch 落在不同序列上，每次都要重新打开
memmap、解压 boxes.npz，CPU 成为瓶颈（实测 smoke 阶段 GPU 利用率 0%）。

这里改成：每个 epoch 打乱"序列的顺序"，但同一序列的 clip 连续取出。
这样一条序列只加载一次，后续全是内存命中。
"""

from __future__ import annotations

import random
from collections import defaultdict

from torch.utils.data import Sampler


class SeqShuffleSampler(Sampler[int]):
    def __init__(self, items: list[tuple[str, int]], seed: int = 42) -> None:
        self.by_seq: dict[str, list[int]] = defaultdict(list)
        for i, (seq, _) in enumerate(items):
            self.by_seq[seq].append(i)
        self.seqs = list(self.by_seq)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = self.seqs[:]
        rng.shuffle(order)
        for s in order:
            idx = self.by_seq[s][:]
            rng.shuffle(idx)
            yield from idx

    def __len__(self) -> int:
        return len(self.by_seq) and sum(len(v) for v in self.by_seq.values())
