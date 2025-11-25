"""
Copyright 2022 Balacoon

feeding models at training/inference
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from balacoon_train.config import Config
from balacoon_train.data.bucketing_batch_sampler import BucketingBatchSampler
from balacoon_train.data.dataset import Dataset, DatasetConfig
from balacoon_train.data.seq_len_readers.sequence_length_reader import SequenceLengthReaderConfig
from balacoon_train.data.sequence_length_collector import SequenceLengthCollector
from balacoon_train.data.validator import Validator


@dataclass
class DataConfig:
    locations: Dict[str, Any] = field(
        default_factory=lambda: {}
    )  #: quick access args, that contain locations of data (data dirs, uttid files)
    speakers: List[str] = field(
        default_factory=lambda: []
    )  #: quick access list of speaker regexes for the given dataset
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig
    )  # defines loading of the data
    seq_len: SequenceLengthReaderConfig = field(
        default_factory=SequenceLengthReaderConfig
    )  # defines how to get sequence lengths for bucketing
    batch_size: int = 32  # batch size to train/inference with
    num_workers: int = 0  # can be changes for single gpu training
    prefetch_factor: int = 2  # how many batches to preload in advance by each worker


def read_ids(path: str) -> List[str]:
    """
    helper function to read
    ids from a file
    """
    assert os.path.isfile(path), "{} does not exist, can't read ids"
    with open(path, "r") as fp:
        ids = [x.strip().split()[0] for x in fp.readlines()]
    return ids


def create_dataset(
    config: Config,
    ids_path: str,
    stride: int = 0,
) -> Dataset:
    """
    creates data loader which produces batches to be used in training/inference.
    native torch data loader uses sampler (defines ids of samples to be used in a batch)
    and dataset (loads the samples of the data).

    Parameters
    ----------
    config: Config
        configuration of the data, contains config for sampler, dataset, sequence length extraction, etc.
    ids_path: str
        path to file with utterance ids to load
    stride: int
        stride of the data loader to return. is used in distributed training

    Returns
    -------
    loader: torch.utils.data.DataLoader:
        native torch loader that can be used by `Trainer` or directly
    """
    config.add_missing(DataConfig)
    stride = 0 if stride < 0 else stride

    valid_ids_path = ids_path + ".valid"
    if os.path.isfile(valid_ids_path):
        # ids are already validated, load them
        time.sleep(1)  # wait for a sec in case file being written by master process
        ids = read_ids(valid_ids_path)
    else:
        # need to validate which utterances can be used for this data loader
        if stride == 0:
            ids = read_ids(ids_path)
            ids = Validator(config.dataset).get_valid_ids(ids)
            # dump validated ids for future training and/or
            # other processes
            with open(valid_ids_path, "w") as fp:
                for name in ids:
                    fp.write(name + "\n")
        else:
            # this is not a master process, wait for master process to validate
            # the utterances
            while not os.path.isfile(valid_ids_path):
                time.sleep(5)
                logging.info(
                    "[{}] waits for master process to validate utterances".format(
                        stride
                    )
                )
            time.sleep(
                1
            )  # wait for just 1 more sec before reading, in case master is writing
            ids = read_ids(valid_ids_path)

    # mapping between indices and utterance names
    idx2id: Dict[str, int] = {i: name for i, name in enumerate(ids)}
    id2idx = {v: k for k, v in idx2id.items()}  # get index to utterance id mapping
    # mapping between indices and sequence lengths
    id2len: Dict[str, int] = {}
    if config.seq_len:
        id2len = SequenceLengthCollector(config.seq_len).get_seq_len(ids)
    else:
        # dummy lengths for ids
        id2len = {x: 1 for x in ids}
    # mapping between utterance indices and their lengths
    idx2len: Dict[int, int] = {id2idx[k]: v for k, v in id2len.items()}
    # create a dataset
    dataset = Dataset(config.dataset, idx2id, idx2len)
    return dataset


def create_data_loader(
    config: Config,
    ids_path: str,
    shuffle: bool = False,
    stride: int = 0,
    strides_num: int = 1,
) -> torch.utils.data.DataLoader:
    dataset = create_dataset(config, ids_path, stride)
    logging.info("Creating bucketing batch sampler (stride: [{}])".format(stride))
    batch_sampler = BucketingBatchSampler(
        dataset.get_idx2len(),
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        stride=stride,
        strides_num=strides_num,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataset.collate,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=False,  # for distributed helps to avoid bugs
    )
    return dataloader


def get_fold_path(config: Config, fold: str) -> Optional[str]:
    """
    getter that implements convention about naming of data folds.
    if fold is absent from config, returns None
    """
    fold_txt_field = fold + "_txt"
    if fold_txt_field not in config.locations:
        logging.warning("There is no {} fold".format(fold))
        return None
    ids_path = config.locations[fold_txt_field]
    return ids_path


def create_fold_data_loader(
    config: Config,
    fold: str,
    shuffle: bool = False,
    stride: int = 0,
    strides_num: int = 1,
) -> Optional[torch.utils.data.DataLoader]:
    """
    Wrapper around `create_data_loader` which derives path to utterance ids file
    from a fold name (train/test/dev). Check above for parameters description
    """
    ids_path = get_fold_path(config, fold)
    if not ids_path:
        return None
    stride = 0 if stride < 0 else stride
    return create_data_loader(
        config, ids_path, shuffle, stride=stride, strides_num=strides_num
    )
