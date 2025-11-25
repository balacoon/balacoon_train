"""
Copyright 2022 Balacoon

functions to help parallel preprocessing of dataset
"""

import multiprocessing
from typing import Any, Callable, List


def split_ids_to_subsets(ids: List[str]) -> List[List[str]]:
    """
    a function used to split ids into subsets, based in nproc, i.e.
    maximum number of processes that machine can run.
    those subsets can be used for parallel validation or sequence length calculation.

    Returns
    -------
    subsets: List[List[str]]
        list of lists of utterance ids. each list is a subset,
        which can be processed separately. subsets are roughly
        equal in size.
    """
    # define number of processes for parallel processing
    nproc = min(multiprocessing.cpu_count(), len(ids), 16)
    k, m = divmod(len(ids), nproc)
    subsets = [ids[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(nproc)]
    return subsets


def parallel_execution(helper: Callable, subsets: List[List[str]]) -> List[Any]:
    """
    a function that spawns a pull of processes and passes each subset
    to a helper in a separate process. Results are collected and returned

    Parameters
    ----------
    helper: Callable
        parallel execution helper which is called multiple times in separate
        processes
    subsets: List[List[str]]
        subsets of utterance ids to process in parallel

    Returns
    -------
    results: List[Any]
        collects from separate processes what helper returns.
        it returns one result per process, i.e. one result per input subset.
    """
    # https://pythonspeed.com/articles/python-multiprocessing/
    with multiprocessing.get_context("spawn").Pool(len(subsets)) as pool:
        results = pool.map(helper, subsets)

    return results
