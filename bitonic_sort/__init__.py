from __future__ import division
import logging

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
        padding = '    ' * self.depth 
        logging.info('i=%d j=%d direction=%d' % (i, j, direction))
        #logging.info(self.data)

    def sort(self, data):
        self.depth = -1
        self.data = data
        self._bitonic_sort(0, len(data), self.ASCENDING)


class BitonicSorterForArbitraryN(BitonicSorter):
    def _bitonic_sort(self, lo, n, direction):
        self.depth += 1
        padding = '    ' * self.depth 
        logging.debug(padding + '[_bitonic_sort:BEGIN] %d, %d, %d' % (lo, n, direction))
        if n > 1:
            m = n // 2
            #self._bitonic_sort(lo, m, not direction)
            #self._bitonic_sort(lo + m, n - m, direction)
            #self._bitonic_sort(lo, n - m, not direction)
            #self._bitonic_sort(lo + n - m, m, direction)
            logging.debug(direction)
            self._bitonic_sort(lo, n - m, not direction)
            self._bitonic_sort(lo + n - m, m, direction)
            self._bitonic_merge(lo, n, direction)
        logging.debug(padding + '[_bitonic_sort:END]   %d, %d, %d' % (lo, n, direction))
        self.depth -= 1

    def _bitonic_merge(self, lo, n, direction):
        self.depth += 1
        padding = '    ' * self.depth 
        logging.debug(padding + '[_bitonic_merge:START] %d, %d, %d' % (lo, n, direction))
        if n > 1:
            m = self._greatest_power_of_two_less_than(n)
            for i in range(lo, lo + n - m):
                self._compare(i, i + m, direction);
            self._bitonic_merge(lo, m, direction);
            self._bitonic_merge(lo + m, n - m, direction);
        logging.debug(padding + '[_bitonic_merge:END]   %d, %d, %d' % (lo, n, direction))
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


class BitonicSorterForArbitraryN_iterative(BitonicSorter):
    def sort(self, data):
        two_power_size = 1 << int(np.log2(len(data)))
        sorter = BitonicSorter()

        # Sort the first largest possible set of elements from the data
        # whose size is a power of two using bitonic sort.
        two_power_data = data[:two_power_size]
        sorter.sort(two_power_data)

        processed = two_power_size
        remaining = data[two_power_size:]

        # Perform a simple insertion sort to insert remaining elements
        for remaining_index in range(len(remaining)):
            compare_value = remaining[0]
            for i in range(processed, -1, -1):
                try:
                    if i == 0 and data[i] > compare_value:
                        data[i] = compare_value
                    elif i > 0:
                        if data[i - 1] >= compare_value:
                            data[i] = data[i - 1]
                        elif data[i] >= compare_value:
                            data[i] = compare_value
                except IndexError:
                    print '%d is not within [0, %d)' % (i, len(data))
                    raise
            remaining = remaining[1:]
            processed += 1
        data_copy = sorted(data)
        assert((data_copy == data).all())
        return data

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
