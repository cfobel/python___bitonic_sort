#ifndef ___BITONIC_SORT__HPP___
#define ___BITONIC_SORT__HPP___

namespace bitonic_sort {
    template <class T>
    __device__ inline void Comparator(
        volatile T& keyA,
        volatile T& keyB,
        bool direction
    ){
        T t;
        if((keyA > keyB) == direction){
            t = keyA;
            keyA = keyB;
            keyB = t;
        }
    }


    #include <stdio.h>
    #include <stdint.h>
    #include <math.h>

    template <class T>
    __device__ void dump_data(int size, volatile T *data) {
        syncthreads();
        if(threadIdx.x == 0) {
            printf("[");
            for(int i = 0; i < size; i++) {
                printf("%d, ", data[i]);
            }
            printf("]\n");
        }
    }


    template <class T>
    __device__ void bitonic_sort(int size, volatile T *data, bool direction) {
        if(threadIdx.x == 0) {
            printf("data = [\n");
        }
        for(uint slice_size = 2; slice_size < size; slice_size <<= 1){
            //Bitonic merge
            uint ddd = direction ^ ( (threadIdx.x & (slice_size / 2)) != 0 );
            for(uint stride = slice_size / 2; stride > 0; stride >>= 1){
                __syncthreads();
                uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
                if(pos + stride < size) {
                    Comparator<T>(data[pos + 0], data[pos + stride], ddd);
#if 0
                } else {
                    printf("dict(slice_size=%d, thread_id=%d, stride=%d, pos=%d, pos_plus_stride=%d),\n",
                            slice_size, threadIdx.x, stride,
                            pos, (pos + stride));
#endif
                }
            }
        }

#if 0
        __syncthreads();
        if(threadIdx.x == 0) {
            printf("]\ndata.sort(key=lambda x: (x['slice_size'], x['thread_id'], x['stride'], x['pos'], x['pos_plus_stride']))\n");
        }
#endif

        //ddd == direction for the last bitonic merge step
        {
            for(uint stride = size / 2; stride > 0; stride >>= 1) {
                __syncthreads();
                uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
                if(pos + stride < size) {
                    Comparator<T>(data[pos + 0], data[pos + stride], direction);
                }
            }
        }

        __syncthreads();
    }
}

#endif
