#include "bitonic_sort.hpp"

{% if not c_types -%}
{%- set c_types=["float", "int"] -%}
{%- endif -%}

{% for c_type in c_types -%}
extern "C" __global__ void bitonic_sort_{{ c_type }}(int size, {{ c_type }} *data) {
    bitonic_sort::bitonic_sort<{{ c_type }}>(size, data);
}
{% endfor %}
