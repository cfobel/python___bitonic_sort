from __future__ import division

import numpy as np


class BitonicSorter(object):
    DESCENDING = False
    ASCENDING = True

    def _bitonic_sort(self, lo, n, direction):
        if n > 1:
            m = n // 2
            self._bitonic_sort(lo, m, self.ASCENDING)
            self._bitonic_sort(lo + m, m, self.DESCENDING)
            self._bitonic_merge(lo, n, direction)

    def _bitonic_merge(self, lo, n, direction):
        if n > 1:
            m = n // 2
            for i in range(lo, lo + m):
                self._compare(i, i + m, direction)
            self._bitonic_merge(lo, m, direction)
            self._bitonic_merge(lo + m, m, direction)

    def _compare(self, i, j, direction):
        if direction == (self.data[i] > self.data[j]):
            self.data[i], self.data[j] = self.data[j], self.data[i]

    def sort(self, data):
        self.depth = -1
        self.data = data
        self._bitonic_sort(0, len(data), self.ASCENDING)


class BitonicSorterForArbitraryN(BitonicSorter):
    def _bitonic_sort(self, lo, n, direction):
        self.depth += 1
        padding = '    ' * self.depth 
        print padding + '[_bitonic_sort:BEGIN] %d, %d, %d' % (lo, n, direction)
        if n > 1:
            m = n // 2
            self._bitonic_sort(lo, m, not direction)
            self._bitonic_sort(lo + m, n - m, direction)
            self._bitonic_merge(lo, n, direction)
        print padding + '[_bitonic_sort:END]   %d, %d, %d' % (lo, n, direction)
        self.depth -= 1

    def _bitonic_merge(self, lo, n, direction):
        self.depth += 1
        padding = '    ' * self.depth 
        print padding + '[_bitonic_merge:START] %d, %d, %d' % (lo, n, direction)
        if n > 1:
            m = self._greatest_power_of_two_less_than(n)
            for i in range(lo, lo + n - m):
                self._compare(i, i + m, direction);
            self._bitonic_merge(lo, m, direction);
            self._bitonic_merge(lo + m, n - m, direction);
        print padding + '[_bitonic_merge:END]   %d, %d, %d' % (lo, n, direction)
        self.depth -= 1

    def _greatest_power_of_two_less_than(self, n):
        '''
        Equivalent to:

        >>> value = np.log2(n)
        >>> if value == int(value):
        ...     value = int(value) - 1
        ... else:
        ...     value = int(value)
        ... print 1 << value
        '''
        k = 1
        while k < n:
            k = k << 1
        return k >> 1


if __name__ == '__main__':
    #np.random.seed(0)
    #data1 = np.random.randint(0, , size=16)
    data1 = np.arange(0, 13)
    data1 = data1[::-1]
    #np.random.shuffle(data1)
    data2 = data1.copy()

    data1.sort()
    #sorter = BitonicSorter()
    sorter = BitonicSorterForArbitraryN()
    sorter.sort(data2)
    conflicts = np.where(data1 != data2)
