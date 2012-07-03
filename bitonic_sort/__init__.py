from __future__ import division

'''
public class BitonicSorter implements Sorter
{
    private int[] a;
    // sorting direction:
    private final static boolean ASCENDING=true, DESCENDING=false;

    public void sort(int[] a)
    {
        this.a=a;
        bitonicSort(0, a.length, ASCENDING);
    }

    private void bitonicSort(int lo, int n, boolean dir)
    {
        if (n>1)
        {
            int m=n/2;
            bitonicSort(lo, m, ASCENDING);
            bitonicSort(lo+m, m, DESCENDING);
            bitonicMerge(lo, n, dir);
        }
    }

    private void bitonicMerge(int lo, int n, boolean dir)
    {
        if (n>1)
        {
            int m=n/2;
            for (int i=lo; i<lo+m; i++)
                compare(i, i+m, dir);
            bitonicMerge(lo, m, dir);
            bitonicMerge(lo+m, m, dir);
        }
    }

    private void compare(int i, int j, boolean dir)
    {
        if (dir==(a[i]>a[j]))
            exchange(i, j);
    }

    private void exchange(int i, int j)
    {
        int t=a[i];
        a[i]=a[j];
        a[j]=t;
    }

}    // end class BitonicSorter
'''

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
            self._exchange(i, j)

    def _exchange(self, i, j):
        temp = self.data[i]
        self.data[i] = self.data[j]
        self.data[j] = temp
        #self.data[i], self.data[j] = self.data[j], self.data[i]

    def sort(self, data):
        self.data = data
        self._bitonic_sort(0, len(data), self.ASCENDING)


class BitonicSorterForArbitraryN(BitonicSorter):
    def _bitonic_sort(self, lo, n, direction):
        if n > 1:
            m = n // 2
            self._bitonic_sort(lo, m, not direction)
            self._bitonic_sort(lo + m, n - m, direction)
            self._bitonic_merge(lo, n, direction)

    def _bitonic_merge(self, lo, n, direction):
        if n > 1:
            m = self._greatest_power_of_two_less_than(n)
            for i in range(lo, lo + n - m):
                self._compare(i, i + m, direction);
            self._bitonic_merge(lo, m, direction);
            self._bitonic_merge(lo + m, n - m, direction);

    def _greatest_power_of_two_less_than(self, n):
        k = 1
        while k < n:
            k = k << 1
        return k >> 1
