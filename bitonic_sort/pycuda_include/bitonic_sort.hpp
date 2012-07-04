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
    __device__ void bitonic_sort(int size, volatile T *data, bool direction) {
#if ___BITONIC_DEBUG___
        if(threadIdx.x == 0) {
            printf("data = [\n");
        }
#endif
        int even_size = size / 2;
        for(uint slice_size = 2; slice_size < even_size; slice_size <<= 1){
            //Bitonic merge
            /*
             * ddd
             * True if ?
             */
            uint ddd = direction ^ ((threadIdx.x & (slice_size / 2)) != 0 );
            data[threadIdx.x] = ddd;
            __syncthreads();
            if(threadIdx.x == 0) printf("Direction:  ");
            dump_data(size, data);

//            uint m = greatest_power_of_two_less_than(slice_size);
            for(uint stride = slice_size / 2; stride > 0; stride >>= 1){
            //for(uint stride = greatest_power_of_two_less_than(slice_size);
            //        stride > 0; stride >>= 1) {
                __syncthreads();
                //uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
                uint pos = 2 * threadIdx.x - (threadIdx.x % stride);
                if(pos + stride < size) {
                    data[pos] = threadIdx.x;
                    data[pos + stride] = threadIdx.x;
                }
                __syncthreads();
                if(threadIdx.x == 0) printf("Thread IDs: ");
                dump_data(size, data);
                //if(pos < slice_size - m && pos + stride < size) {
#if 0
                if(pos + stride < size) {
                    Comparator<T>(data[pos + 0], data[pos + stride], ddd);
#if ___BITONIC_DEBUG___
                } else {
                    printf("dict(slice_size=%d, thread_id=%d, stride=%d, pos=%d, pos_plus_stride=%d),\n",
                            slice_size, threadIdx.x, stride,
                            pos, (pos + stride));
#endif
                }
#endif
            }
        }

#if ___BITONIC_DEBUG___
        __syncthreads();
        if(threadIdx.x == 0) {
            printf("]\ndata.sort(key=lambda x: (x['slice_size'], x['thread_id'], x['stride'], x['pos'], x['pos_plus_stride']))\n");
        }
#endif

        data[threadIdx.x] = direction;
        __syncthreads();
        if(threadIdx.x == 0) printf("Direction:  ");
        dump_data(size, data);
        //ddd == direction for the last bitonic merge step
        {
            for(uint stride = size / 2; stride > 0; stride >>= 1) {
                __syncthreads();
                uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
#if 0
                if(pos + stride < size) {
                    Comparator<T>(data[pos + 0], data[pos + stride], direction);
                }
#endif
                if(pos + stride < size) {
                    data[pos] = threadIdx.x;
                    data[pos + stride] = threadIdx.x;
                }
                __syncthreads();
                if(threadIdx.x == 0) printf("Thread IDs: ");
                dump_data(size, data);
            }
        }

        __syncthreads();
    }
}

#endif
