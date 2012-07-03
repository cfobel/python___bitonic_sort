from __future__ import division
import copy
import math

from nose.tools import eq_, ok_, nottest
import numpy as np

from ..bitonic_sort import BitonicSorter, BitonicSorterForArbitraryN
from ..bitonic_sort.cuda import sort_inplace


@nottest
def test_sort(sort_func, count):
    np.random.seed(0)
    data1 = np.random.randint(0, 100, size=count)
    data2 = data1.copy()

    data1.sort()
    sort_func(data2)
    conflicts = np.where(data1 != data2)
    if conflicts[0]:
        print conflicts
        print data1[conflicts]
        print data2[conflicts]
        raise ValueError


@nottest
def test_bitonic2n():
    sorter = BitonicSorter()
    for i in range(4, 11):
        yield test_sort, sorter.sort, (1 << i)


@nottest
def test_bitonic_any_n():
    sorter = BitonicSorterForArbitraryN()
    for i in range(2, 4):
        yield test_sort, sorter.sort, 10 ** i


def test_bitonic_cuda():
    data = np.arange(10, dtype=np.int32)
    data = data[::-1]
    print sort_inplace(data)
