"""AdaBound for Tensorflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


@tf_export("train.AdaBoundOptimizer")
class AdaBoundOptimizer(optimizer.Optimizer):
    """Optimizer that implements the AdaBound algorithm.

    See [Luo et al., 2019](https://openreview.net/forum?id=Bkg3g2R9FX)
    ([pdf](https://openreview.net/pdf?id=Bkg3g2R9FX)).
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, final_lr=0.1, gamma=1e-3,
                 epsilon=1e-8, weight_decay=0., amsbound=False,
                 use_locking=False, name="AdaBound"):
        super(AdaBoundOptimizer, self).__init__(use_locking, name)

        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._final_lr = final_lr
        self._gamma = gamma
        self._epsilon = epsilon
        self._weight_decay = weight_decay
        self._amsbound = amsbound

        # Tensor versions of the constructor arguments, created in _prepare()
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._final_lr = None
        self._gamma_t = None
        self._epsilon_t = None
        self._weight_decay_t = None
        self._amsbound_t = None

        # Created in SparselyApply if needed
        self._updated_lr = None

    def _prepare(self):
        pass


