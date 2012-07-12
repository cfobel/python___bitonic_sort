from __future__ import division
import copy
import math
import random

from nose.tools import eq_, ok_, nottest
import numpy as np

from ..bitonic_sort import BitonicSorter, BitonicSorterForArbitraryN,\
         BitonicSorterForArbitraryN_iterative
from ..bitonic_sort.cuda import sort_inplace


@nottest
def test_sort(sort_func, count):
    if count == 5:
        data1 = np.array([3, 0, 4, 1, 2])
    else:
        np.random.seed(0)
        data1 = np.random.randint(0, count, size=count)
    data2 = data1.copy()

    print data1
    data1.sort()
    sort_func(data2)
    print data2
    conflicts = np.where(data1 != data2)
    if conflicts[0]:
        print conflicts
        print data1[conflicts]
        print data2[conflicts]
        raise ValueError


def test_bitonic2n():
    sorter = BitonicSorter()
    for i in range(4, 11):
        yield test_sort, sorter.sort, (1 << i)

def test_bitonic_any_n():
    sorter = BitonicSorterForArbitraryN()
    for i in range(2, 4):
        yield test_sort, sorter.sort, 10 ** i


@nottest
def test_bitonic_cuda_two_power_single(gold_data, data, direction, thread_count):
    cuda_data_ = sort_inplace(data, ascending=direction,
            thread_count=thread_count)
    if not (cuda_data == gold_data).all():
        print 'cuda_data:', cuda_data
        print 'gold_data:', gold_data
    ok_((cuda_data == gold_data).all())

@nottest
def test_bitonic_cuda_two_power(data_size, thread_count=None):
    #assert(np.log2(data_size) == int(np.log2(data_size)))

    np.random.seed(0)
    data = np.arange(data_size, dtype=np.int32)
    np.random.shuffle(data)
    data_ascending = data.copy()
    data_ascending.sort()
    data_descending = data_ascending[::-1]
    gold_data = [data_descending, data_ascending]

    if thread_count:
        thread_counts = [thread_count]
    else:
        thread_counts = [1, 3, 7, 32, 47, 129, 138]

    for direction in [True, False]:
        for thread_count in thread_counts:
            yield test_bitonic_cuda_two_power_single,\
                    gold_data[direction], data, thread_count


def test_bitonic_cuda():
    random.seed(0)

    max_power = 11
    trial_sizes = [(1 << power) + random.randint(-5, 5)
            for power in range(3, max_power + 1)]
    trial_sizes = [min(v, 1 << max_power) for v in trial_sizes]

    for n in trial_sizes:
        yield test_bitonic_cuda_two_power(n)
