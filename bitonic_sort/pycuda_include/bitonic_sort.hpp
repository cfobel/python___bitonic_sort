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
        int count = 0;

        int two_power_size = 1 << (int)log2((float)size);
        bool ddd = direction;

        /* Sort the first largest possible set of elements from the data
         * whose size is a power of two using bitonic sort.
         */
        do_bitonic_sort(two_power_size, data, ddd);
        do_bitonic_merge(two_power_size, data, ddd);

        int processed = two_power_size;

        // Perform a simple insertion sort to insert remaining elements
        for(int remaining_index = 0; remaining_index < size - two_power_size;
                remaining_index++) {
            __syncthreads();
            T compare_value = data[processed];
            __syncthreads();
            T temp;
            int passes = ceil((float)(processed + 1) / blockDim.x);
            for(int k = 0; k < passes; k++) {
                int i = processed - (k * blockDim.x + threadIdx.x);
                if(i >= 0) {
                    if(i > 0) {
                        temp = data[i - 1];
#if 0
                        if(i == processed) printf("(processed) i=%d data[%d]\n", i, temp);
#endif
                    }
                }
                __syncthreads();
                if(i >= 0) {
#if 0
                    printf("i=%d temp=%d compare_value=%d\n", i, temp, compare_value);
#endif
                    
                    if(i > 0) {
                        if(temp >= compare_value) {
#if 0
                            printf("temp=%d >= compare_value=%d -> copy left value to data[%d]\n", temp, compare_value, i);
#endif
                            data[i] = temp;
                        } else if(data[i] >= compare_value) {
#if 0
                            printf("data[%d]=%d >= compare_value=%d -> copy compare_value to data[%d]\n", i, data[i], compare_value, i);
#endif
                            data[i] = compare_value;
                        }
                    } else if(i == 0 && data[i] > compare_value) {
#if 0
                        printf("i == 0 and data[%d]=%d > compare_value=%d -> copy compare_value to data[%d]\n", i, data[i], compare_value, i);
#endif
                        data[i] = compare_value;
                    }
#if 0
                    else {
                        printf("data[%d]=%d compare_value=%d \n", i, data[i], compare_value);
                    }
#endif
                }
            }
            processed += 1;
        }
    }
}

#endif
