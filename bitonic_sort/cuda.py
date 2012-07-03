from __future__ import division
import math

from jinja2 import Template, FileSystemLoader, Environment
from path import path
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


def module_root():
    '''
    Return absolute path to pyvpr root directory.
    '''
    try:
        script = path(__file__)
    except NameError:
        script = path(sys.argv[0])
    return script.parent.abspath()


def get_include_root():
    return module_root().joinpath('pycuda_include')


def get_template_root():
    return module_root().joinpath('pycuda_templates')


def get_template_loader():
    return FileSystemLoader(get_template_root())


jinja_env = Environment(loader=get_template_loader())


def sort_inplace(in_data, ascending=True, dtype=None):
    code_template = jinja_env.get_template('bitonic_sort.cu')
    mod = SourceModule(code_template.render(), no_extern_c=True,
            options=['-I%s' % get_include_root()], keep=True)

    if dtype is None:
        dtype = in_data.dtype
    assert(dtype in [np.int32, np.float32])

    dtype_map = { np.dtype('float32'): 'float', np.dtype('int32'): 'int'}

    try:
        func_name = 'bitonic_sort_%s' % dtype_map[dtype]
        test = mod.get_function(func_name)
    except drv.LogicError:
        print dtype, func_name
        raise

    data = np.array(in_data, dtype=dtype)

    shared = len(data) * dtype.itemsize
    block_count = 1

    block = (len(data), 1, 1)
    grid = (block_count, 1, 1)

    test(np.int32(len(data)), drv.InOut(data), np.uint8(ascending), block=block,
            grid=grid, shared=shared)

    return data
