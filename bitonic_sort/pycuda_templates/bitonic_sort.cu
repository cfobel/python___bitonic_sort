#include "bitonic_sort.hpp"

{% if not c_types -%}
{%- set c_types=["float", "int"] -%}
{%- endif -%}

extern __shared__ float shared_data[];

{% for c_type in c_types -%}
extern "C" __global__ void bitonic_sort_{{ c_type }}(int size, {{ c_type }} *data, bool direction) {
    {{ c_type }} *sh_data = ({{ c_type }} *)&shared_data[0];

    if(threadIdx.x < size) {
        sh_data[threadIdx.x] = data[threadIdx.x];
    }

    bitonic_sort::bitonic_sort<{{ c_type }}>(size, &sh_data[0], direction);

    if(threadIdx.x < size) {
        data[threadIdx.x] = sh_data[threadIdx.x];
    }
}
{% endfor %}
