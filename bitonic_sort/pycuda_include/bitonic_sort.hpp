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
    __device__ T greatest_power_of_two_less_than(T n) {
        T k = 1;
        while(k < n) {
            k = k << 1;
        }
        return k >> 1;
    }


    template <class T>
    __device__ void do_bitonic_sort(int size, volatile T *data, bool direction) {
        for(uint slice_size = 2; slice_size < size; slice_size <<= 1){
            //Bitonic merge
            uint ddd = direction ^ ((threadIdx.x & (slice_size / 2)) != 0 );

            for(uint stride = slice_size / 2; stride > 0; stride >>= 1){
            //for(uint stride = greatest_power_of_two_less_than(slice_size);
            //        stride > 0; stride >>= 1) {
                __syncthreads();
                /*
                 * The following line is equivalent to:
                 *     uint pos = 2 * threadIdx.x - (threadIdx.x % stride);
                 */
                uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
                if(pos + stride < size) {
                    Comparator<T>(data[pos + 0], data[pos + stride], ddd);
                }
            }
        }
    }


    template <class T>
    __device__ void do_bitonic_merge(int size, volatile T *data, bool direction) {
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


    template <class T>
    __device__ void do_bitonic_final_merge(int size, volatile T *data, bool direction) {
        if(threadIdx.x == 0 ) {
            printf("[do_bitonic_final_merge] size=%d data=[", size);
            for(int i = 0; i < size; i++) {
                printf("%d, ", data[i]);
            }
            printf("]\n");
        }

        int init_size = size;
        size = 1 << (int)ceil(log2((float)size));
        //ddd == direction for the last bitonic merge step
        {
            for(uint stride = size / 2; stride > 0; stride >>= 1) {
                __syncthreads();
                uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
                if(pos + stride < init_size) {
                    Comparator<T>(data[pos + 0], data[pos + stride], direction);
                }
            }
        }
    }


    template <class T>
    __device__ void bitonic_sort(int size, volatile T *data, bool direction) {
        int processed = 0;

        int two_power_size = 1 << (int)log2((float)size);
        bool ddd = direction;

        int count = 0;
        volatile T *current_data = data;

        while(processed < size) {
            printf("two_power_size=%d\n", two_power_size);
            do_bitonic_sort(two_power_size, current_data, ddd);
            do_bitonic_merge(two_power_size, current_data, ddd);
            ddd = !ddd;
            processed += two_power_size;
            two_power_size = 1 << (int)log2((float)size - processed);
            current_data = &data[processed];
            if(count > 0) {
                do_bitonic_final_merge(current_data - data, data, direction);
            }
            __syncthreads();
            count++;
        }
    }
}

#endif
