#ifndef ___BITONIC_SORT__HPP___
#define ___BITONIC_SORT__HPP___

namespace bitonic_sort {
    #include <stdio.h>
    #include <stdint.h>
    #include <math.h>

    template <class T>
    __device__ void dump_data(int size, T *data) {
        syncthreads();
        if(threadIdx.x == 0) {
            printf("[");
            for(int i = 0; i < size; i++) {
                printf("%g, ", data[i]);
            }
            printf("]\n");
        }
    }


    template <class T>
    __device__ void bitonic_sort(int size, T *data) {
        dump_data<T>(size, data);
    }
}

#endif
