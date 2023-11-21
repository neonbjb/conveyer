# Conveyer
![An image](assets/dalle.png)
Conveyer is a simple library for building data pipelines in Python. It provides the following features:

## Overview

- Determinism: While ML data pipelines must be non-deterministic, we can control the randomness using PRNGs with
               seeds. Conveyer pipelines are intended to be deterministic by default.
- State: the pipeline can be saved to disk and resumed at any time
- Multi-thread, multi-rank: Supports training on multiple GPUs with multiple data loading workers per GPU. All ranks
  get their own data. State is shared across all ranks; restoring one rank synchronizes all ranks.
- Composable: Multiple pipelines can be joined together with weighting factors.

## Usage

Lets say you have a bunch of images in a directory, and you want to set up a pipeline to iterate through all of them on
8 GPUs. Here's how that'd look:

```python
import os
import torch
import torch.distributed as dist
from conveyer import Permutation, Collate, Partition, Conveyer, DistributedBatchedMap

def load_image(path: str) -> dict[str, torch.Tensor]:
    # up to you to implement this.
    return dict(image=None)

if __name__ == '__main__':
    all_files = os.glob('path/to/images/*.jpg')
    pipeline = Conveyer([
        # We want a random permutation of all available files.
        Permutation(len(all_files)),
        # Each GPU gets its own partition of that permutation. This must always occur before distributed work!
        Partition(dist.get_rank(), dist.get_world_size()),
        # Load images in parallel.
        DistributedBatchedMap(load_image, n_workers=4),
        # Collage several images into a batch.
        Collate(batch_size=8),
    ])

    # Grab a batch! Note that every rank gets distinct data.
    batch = pipeline()
    # ... do some training or whatever ...
    state = pipeline.state()
    torch.save(state, "dataset_state.pt")  # The same state is shared across ranks.
    # ... when you need to resume ...)
    pipeline = Conveyer(...)
    pipeline.load_state(torch.load("dataset_state.pt"))
```

Lets say you have several different Conveyers for different datasets, and you want to blend them
together. That's pretty easy:

```python
from conveyer import Join

imagenet = Conveyer(...)
cifar1k = Conveyer(...)
lsun = Conveyer(...)
pipeline = Join([
    (imagenet, .7),
    (lsun, .25),
    (cifar1k, .05),
])
```

The above sets up a pipeline that is 70% imagenet, 25% lsun, and 5% cifar1k. The weights don't have to sum to 1, but
they should always be > .001. In practice, the way to configure weights is to start with the total size of each sub-dataset.
This sets up a pipeline that samples from each sub-dataset in a way that is proportional to their contents, so an entire
epoch will sample all elements from each sub-dataset.

From there, you can bump weights up or down as desired to prioritize certain sub-datasets.