#include "bitonic_sort.hpp"

{% if not c_types -%}
{%- set c_types=["float", "int"] -%}
{%- endif -%}

extern __shared__ float shared_data[];

{% for c_type in c_types -%}
extern "C" __global__ void bitonic_sort_{{ c_type }}(int size, {{ c_type }} *data, bool direction) {
    {{ c_type }} *sh_data = ({{ c_type }} *)&shared_data[0];

    for(int i = threadIdx.x; i < size; i += blockDim.x) {
        sh_data[i] = data[i];
    }

    bitonic_sort::bitonic_sort<{{ c_type }}>(size, &sh_data[0], direction);

    for(int i = threadIdx.x; i < size; i += blockDim.x) {
        data[i] = sh_data[i];
    }
}
{% endfor %}
