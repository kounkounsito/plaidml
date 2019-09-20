# Copyright 2019, Intel Corporation

# Generate the training data for machine learning-based cost model

import plaidml

import functools
import numpy as np
import operator
import os
import sys
import time

from plaidml.keras import backend as pkb
from keras.backend import floatx

def m(*args, **kwargs):
  dtype = kwargs.get('dtype', floatx())
  """Makes a test matrix whose dimensions are the supplied arguments."""
  total = functools.reduce(operator.mul, args, 1)
  arr = np.array(range(-2, total - 2), dtype=dtype)
  if (arr.dtype in [
            "float16",
            "float32",
            "float64",
  ]):
    arr = arr / 2.0
  arr = np.reshape(arr, args)
  return arr

def _conv_inp(IN, IC, OC, IS, KS, strides, padding = 'valid', data_format = None):
    kernel_mat_np = m(*(KS + [IC, OC]))
    if data_format == 'channels_first':
        input_mat_np = m(*([IN] + [IC] + IS))
    else:
        input_mat_np = m(*([IN] + IS + [IC]))
    inputMat = input_mat_np
    kernelMat = kernel_mat_np
    return [inputMat, kernelMat, {'strides': strides, 'padding': padding, 'data_format': data_format}]

class TilePlanRunner(object):
  def run_plaidml_backend(self,
                          data,
                          test_func,
                          backend,
                          num_iterations,
                          dtype,
                          shapes):
    if shapes:
      x = [backend.placeholder(shape = t) for t in shapes]
    else:
      x = [backend.placeholder(shape = t.shape) for t in data if hasattr(t, 'shape')]
    xv = [backend.variable(t, dtype = dtype) for t in data if hasattr(t, 'shape')]
    ps = [t for t in data if not hasattr(t, 'shape')]
    funcs = test_func(*(xv), **ps[0])
    tot_time = 0
    result = []
    for it_counter in range(num_iterations):
      start = time.time()
      # Evaluate forward operation
      result = funcs.eval()
      end = time.time()
      tot_time += (end - start)
    tot_time /= num_iterations
    print("    Testing took: %s sec." % (tot_time))
    return tot_time

  def run(self,
          test_func,
          in_data,
          input_shapes = None,
          dtype = floatx(),
          num_iterations = 1):
    exec_times = []
    count = 0;
    for didx, data in enumerate(in_data):
      shapes = None
      if input_shapes:
        shapes = input_shapes[didx]
      count += 1
      print('    running: {}/{}'.format(count, len(in_data)))
      sys.stdout.flush()
      exec_time = self.run_plaidml_backend(data,
                                           test_func,
                                           pkb,
                                           num_iterations,
                                           dtype,
                                           shapes)
      exec_times.append(exec_time)
    return exec_times

class TilePlanGenerator(object):
  def __init__(self, backend):
    self.backend_ = backend
    self.runner_ = TilePlanRunner()

  def testDot(self):
    data = [
      [m(64, 64), m(64, 64), {}],
      [m(512, 512), m(512, 512), {}],
      [m(1024, 1024), m(1024, 1024), {}],
      [m(2048, 2048), m(2048, 2048), {}],
    ]
    return self.runner_.run(self.backend_.dot, data)

  def testConv1d(self):
    data = [
      _conv_inp(IN = 1, IC = 3, OC = 1, IS = [8], KS = [2],
          strides = (1), padding = 'valid', data_format = 'channels_last'),
      _conv_inp(IN = 2, IC = 1, OC = 4, IS = [8], KS = [3],
          strides = (1), padding = 'valid', data_format = 'channels_last'),
    ]
    return self.runner_.run(self.backend_.conv1d, data)

  def testConv2d(self):
    data = [
      _conv_inp(IN = 3, IC = 3, OC = 1, IS = [9, 8], KS = [2, 2],
          strides = (1, 1), padding = 'valid', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 1, OC = 3, IS = [5, 4], KS = [3, 3],
          strides = (1, 1), padding = 'valid', data_format = 'channels_first'),
      _conv_inp(IN = 2, IC = 4, OC = 2, IS = [5, 5], KS = [2, 2],
          strides = (1, 1), padding = 'valid', data_format = 'channels_first'),
    ]
    return self.runner_.run(self.backend_.conv2d, data)

  def runAll(self):
    for func in dir(self):
      if func.startswith('test'):
        to_call = getattr(self, func)
        print('Testing ', func)
        sys.stdout.flush()
        to_call()

if __name__ == '__main__':
  if 'TEST_OUTPUT' not in os.environ:
    print('Need environment variable TEST_OUTPUT for the filename of the test result')
    exit()
  out_fn = os.environ['TEST_OUTPUT']
  if os.path.exists(out_fn):
    os.remove(out_fn)
  ttp = TilePlanGenerator(pkb)
  ttp.runAll()
