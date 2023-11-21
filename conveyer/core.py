from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import torch
from pygfc import Permutation as GfcPermutation


class Stage:
    """
    Stage base-class.
    """
    _input_stage = None

    def state(self):
        """ Get the current state of this stage and predecessors. Most stages don't have a state, and should return the parent."""
        return self._input_stage.state()

    def set_state(self, state):
        """ Restores the state of this stage and predecessors."""
        self._input_stage.set_state(state)


@dataclass
class Range(Stage):
    """
    Initial stage that emits integers between [0, n_elements).
    If repeating=True, the stage will repeat the sequence indefinitely.
    """
    n_elements: int
    repeating: bool = True
    _current_index = 0

    def __call__(self):
        self._current_index += 1
        if self._current_index > self.n_elements:
            if self.repeating:
                self._current_index = 1
            else:
                raise StopIteration
        return self._current_index - 1

    def state(self):
        return dict(type="Range", current_index=self._current_index)

    def set_state(self, state):
        assert state['type'] == 'Range'
        self._current_index = state['current_index']


@dataclass
class Permutation(Stage):
    """
    Initial stage that emits a random permutation of integers between [0, n_elements).
    If repeating=True, the stage will repeat the sequence indefinitely.
    seed controls the PRNG seed that generates the sequence
    rounds is the number of passes of the Feistel permutation, which can affect randomness.
    """
    n_elements: int
    seed: int = 1234
    rounds: int = 8
    repeating: bool = True
    _current_index = 0

    def __post_init__(self):
        self.permutation = GfcPermutation(self.seed, self.n_elements, self.rounds)

    def __call__(self):
        if self._current_index == self.n_elements:
            raise StopIteration
        res = self.permutation[self._current_index]
        self._current_index += 1
        if self.repeating and self._current_index == self.n_elements:
            self._current_index = 0
        elif self._current_index >= self.n_elements:
            raise StopIteration
        return res

    def state(self):
        return dict(type='Permutation', current_index=self._current_index)

    def set_state(self, state):
        assert state['type'] == 'Permutation'
        self._current_index = state['current_index']


@dataclass
class Join(Stage):
    """
    This is an initial stage which joins together multiple previous conveyers.

    Input is a list of conveyers along with weighting factors.

    Every time this stage is called, a psuedo-random conveyer is selected and called, using the weighting factors
    provided. Note that the resolution of the weighting factors is 1e-3.

    Uses the same random-access PRNG as Permutation to ensure the order of sampling is always the same.
    """
    stages: List[Tuple[Callable, float]]
    seed: int = 4321

    def __post_init__(self):
        self._weights = [int(s[1] * 1000) for s in self.stages]
        self._total_weights = sum(self._weights)
        self._current_index = 0
        self._perm = GfcPermutation(self._total_weights, 8, self.seed)

    def __call__(self):
        idx = self._perm[self._current_index]
        self._current_index = (self._current_index + 1) % self._total_weights
        for i, w in enumerate(self._weights):
            if idx < w:
                return self.stages[i][0]()
            idx -= w
        raise RuntimeError('Should not be able to get here')

    def state(self):
        return {'type': 'Join', 'previous_stages': [s.state() for s in self.stages]}

    def set_state(self, state):
        assert state['type'] == 'Join'
        for i, s in enumerate(self.stages):
            s.set_state(state['previous_stages'][i])


@dataclass
class Partition(Stage):
    """
    Shards the previous stage across multiple concurrent ranks. This is for distributed training across multiple GPUs.
    Sharding should happen early in the pipeline, before any expensive dataloading; ideally immediately after the
    initial stage.
    """
    rank: int
    n_ranks: int

    def __post_init__(self):
        self._input_stage: Callable = None

    def __call__(self):
        for _ in range(self.rank):
            _ = self._input_stage()
        v = self._input_stage()
        for _ in range(self.n_ranks - self.rank - 1):
            _ = self._input_stage()
        return v

    def state(self):
        return self._input_stage.state()

    def set_state(self, state):
        self._input_stage.set_state(state)


@dataclass
class Collate(Stage):
    """
    Converts np arrays to torch tensors and collates torch tensors across the batch dimension.
    - Assumes the output of the previous stage is a dict.
    - Dict can have any values but only np.ndarray and torch.tensors are processed.
    if return_partial_batch=True and the previous stage aborts early, the collate stage will return a partial batch.
    """
    batch_size: int
    return_partial_batch: bool = True

    def __post_init__(self):
        assert self.batch_size > 0, f"batch_size must be > 0, got {self.batch_size}"
        self._input_stage: Callable = None

    def __call__(self):
        batch = []
        for _ in range(self.batch_size):
            try:
                batch.append(self._input_stage())
            except StopIteration:
                if not self.return_partial_batch:
                    raise
                break
        if len(batch) == 0:
            raise StopIteration
        out = {}
        for k, v in batch[0].items():
            if isinstance(v, torch.Tensor):
                stacked = torch.stack([b[k] for b in batch])
            elif isinstance(v, np.ndarray):
                stacked = torch.from_numpy(np.stack([b[k] for b in batch]))
            elif isinstance(v, (int, float)):
                stacked = torch.tensor([b[k] for b in batch])
            else:
                stacked = [b[k] for b in batch]
            out[k] = stacked
        return out

    def state(self):
        return self._input_stage.state()

    def set_state(self, state):
        self._input_stage.set_state(state)


@dataclass
class Map(Stage):
    """
    Applies a function to the output of the previous stage.
    """
    fn: Callable

    def __post_init__(self):
        self._input_stage: Callable = None

    def __call__(self):
        return self.fn(self._input_stage())

    def state(self):
        return self._input_stage.state()

    def set_state(self, state):
        self._input_stage.set_state(state)


@dataclass
class Conveyer(Stage):
    """
    Main class used to join joining together multiple conveyer stages.
    """
    stages: List[Callable]

    def __post_init__(self):
        for i in range(len(self.stages)):
            if i == 0:
                continue
            TypesOfInitialStages = (Range, Permutation, Join)
            for typ in TypesOfInitialStages:
                assert not isinstance(self.stages[i], typ), "Initial stages can only be at the beginning of a conveyer!"
            self.stages[i]._input_stage = self.stages[i - 1]

    def __call__(self):
        return self.stages[-1]()

    def __iter__(self):
        return self

    def __next__(self):
        return self()

    def close(self):
        for stage in self.stages[::-1]:
            if hasattr(stage, 'close'):
                stage.close()

    def state(self):
        return self.stages[-1].state()

    def set_state(self, state):
        self.stages[-1].set_state(state)
