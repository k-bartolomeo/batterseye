import random

from typing import Generator


class PairGenerator:
    def __init__(self, pairs, labels, training: bool = False):
        self.pairs = pairs
        self.labels = labels
        self.training = training

    def __len__(self):
        return len(self.pairs)
    
    def __call__(self) -> Generator[tuple[dict, int]]:
        pairs = list(zip(self.pairs, self.labels))
        if self.training:
            random.shuffle(pairs)
        for i in range(len(pairs)):
            imgs = pairs[i][0]
            label = pairs[i][1]
            yield {'x1': imgs[0], 'x2': imgs[1]}, label

