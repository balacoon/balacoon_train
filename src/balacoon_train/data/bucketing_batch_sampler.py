"""
Copyright 2022 Balacoon

bucketing batch sampler creates batches
"""

import math
from typing import Dict, Iterator, List

import torch


class BucketingBatchSampler(
    torch.utils.data.distributed.DistributedSampler, torch.utils.data.BatchSampler
):
    """
    Similar to native Batchsampler - selects sample indices,
    aggregates them into a batch. `bucketing` means that
    samples for a batch are not selected randomly, but from
    a single bucket. Bucket contains samples with a similar
    sequence length, so when samples are collated, there is
    not a lot of padding to do. This reduces stohasticity of training
    but speeds it up.

    To implement a sampler interface, class implements `__iter__` method,
    and generally follows logic from:
    https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#BatchSampler

    Class is suitable to be used in distributed training, where unique subset of batches is reserved
    for each process (gpu). Implementation follows
    https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
    Object inherits distributed sampler but doesn't call parent constructor, all the computation
    happens explicitely for visibility.
    """

    def __init__(
        self,
        idx2len: Dict[int, int],
        batch_size: int,
        bucket_size_multiplier: int = 10,
        shuffle: bool = True,
        drop_last: bool = True,
        stride: int = 0,
        strides_num: int = 1,
    ):
        """
        constructor of bucketing batch sampler

        Parameters
        ----------
        idx2len: Dict[int, int]
            mapping between sample idx and sample length.
            sampler returns indexes. length is used for bucketing.
        batch_size: int
            how many samples to return per batch
        bucket_size_multiplier: int
            together with `batch_size` defines sizes of the buckets,
            which is `bucket_size_multiplier * batch_size`
        shuffle: bool
            whether to shuffle the input, typically should be
            enabled justs for training
        drop_last: bool
            whether to drop any left-overs from bucketing/batching.
            if enabled (usually for train/dev) not all the samples
            would be yielded. should be disabled for testing,
            where we want to attend ALL the samples. It is achieved
            by repeating samples to get dataset of divisible size.
        stride: int
            stride of the dataset. used in distributed training, where
            one stride is used per process (GPU)
        strides_num: int
            total number of strides into which the dataset should be split
            into. Used in distributed training.
        """
        self.batch_size = batch_size
        self._bucket_size = bucket_size_multiplier * batch_size
        self._buckets = self._split_into_buckets(idx2len)
        assert (
            len(self._buckets) > 0
        ), "No buckets where created. There were {} samples passed".format(len(idx2len))
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.stride = stride
        self.strides_num = strides_num

        # compute how many batches there are in buckets
        samples_num = len(idx2len)
        batch_num = samples_num / float(self.batch_size)
        # number of batches depends if we drop left overs or add extra
        batch_num = math.floor(batch_num) if self.drop_last else math.ceil(batch_num)
        batch_num = int(batch_num)
        # number of batches should evenly devide by number of processes
        extra = batch_num % self.strides_num
        if extra:
            batch_num += self.strides_num - extra
        self.batch_num_per_stride = batch_num // self.strides_num
        self.batch_num = batch_num

        # epoch is used as a seed in case batches are shuffled
        self.epoch = 0

    def set_epoch(self, epoch: int):
        """
        Sets epoch for sampler. Is called from trainer and is used
        as a seed to random function.
        """
        self.epoch = epoch

    def _split_into_buckets(self, idx2len: Dict[int, int]) -> List[List[int]]:
        """
        Splits sample indices into buckets based on their lengths.
        buckets have fixed size of `self._bucket_size`. Last bucket
        has remaining samples. If remaining samples are less than a batch size,
        they are appended to last bucket.

        Parameters
        ----------
        idx2len: Dict[int, int]
            mapping between sample id and sample length

        Returns
        -------
        buckets: List[List[int]]
            list of buckets, where each bucket - is a list of sample ids, belonging to that bucket.
        """
        # https://github.com/python/mypy/issues/9765
        sorted_idxs = sorted(idx2len, key=idx2len.__getitem__)
        buckets = []
        # num of buckets not counting potentially the last not full bucket.
        num_buckets = len(sorted_idxs) // self._bucket_size
        for i in range(num_buckets):
            b = sorted_idxs[i * self._bucket_size : (i + 1) * self._bucket_size]
            buckets.append(b)
        if len(sorted_idxs) % self._bucket_size > 0:
            remaining_idxs = sorted_idxs[num_buckets * self._bucket_size :]
            if not buckets or len(remaining_idxs) > self.batch_size:
                buckets.append(remaining_idxs)
            else:
                buckets[-1].extend(remaining_idxs)
        return buckets

    def __iter__(self) -> Iterator[List[int]]:
        # similar to https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
        # create epoch dependent random generator. it is used only if shuffle is true
        generator = torch.Generator()
        generator.manual_seed(self.epoch)

        # shuffle elements inside of buckets if this is enabled
        buckets = []
        for b in self._buckets:
            if self.shuffle:
                indices = torch.randperm(len(b), generator=generator).tolist()
            else:
                indices = list(range(len(b)))
            shuffled_b = [b[j] for j in indices]
            buckets.append(shuffled_b)

        # go through shuffled buckets and create batches
        batches = []
        for b in buckets:
            for i in range(0, len(b), self.batch_size):
                batches.append(b[i : (i + self.batch_size)])
        # drop last batch if its smaller than batch_size
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches.pop()

        # shuffle batches, so batches from different buckets
        # are next to each other
        if self.shuffle:
            indices = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in indices]

        # if distributed training, make number of batches divisible by number of processes
        if self.strides_num > 1:
            extra = len(batches) % self.strides_num
            if extra != 0:
                # need to repeat some batches
                to_add = self.strides_num - extra
                batches.extend(batches[:to_add])
        assert (
            len(batches) == self.batch_num
        ), "actual number of batches {}, calculated: {}, drop last {}, shuffle {}, abtch size {}".format(
            len(batches), self.batch_num, self.drop_last, self.shuffle, self.batch_size
        )

        # select batches for particular process
        # if strides_num==1 and stride==0, statement has no effect
        batches = batches[self.stride :: self.strides_num]
        assert len(batches) == self.batch_num_per_stride

        return iter(batches)

    def __len__(self) -> int:
        """
        returns number of batches that sampler can generate
        for given data (per GPU process)
        """
        return self.batch_num_per_stride
