"""AdaBound for Tensorflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export


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

    def _get_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return (self._get_non_slot_variable("beta1_power", graph=graph),
                    self._get_non_slot_variable("beta2_power", graph=graph),
                    self._get_non_slot_variable("gamma_power", graph=graph))

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=self._beta1,
                                       name="beta1_power",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._beta2,
                                       name="beta2_power",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._gamma,
                                       name="gamma_power",
                                       colocate_with=first_var)

        # Create slots for the first and second moments
        for v in var_list:
            self._zeros_slot(v, "exp_avg", self._name)
            self._zeros_slot(v, "exp_avg_sq", self._name)
            if self._amsbound:
                self._zeros_slot(v, "max_exp_avg_sq", self._name)

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        final_lr = self._call_if_callable(self._final_lr)
        gamma = self._call_if_callable(self._gamma)
        epsilon = self._call_if_callable(self._epsilon)
        weight_decay = self._call_if_callable(self._weight_decay)
        amsbound = self._call_if_callable(self._amsbound)

        self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._final_lr = ops.convert_to_tensor(final_lr, name="final_lr")
        self._gamma_t = ops.convert_to_tensor(gamma, name="gamma")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")
        self._weight_decay_t = ops.convert_to_tensor(weight_decay, name="weight_decay")
        self._amsbound_t = ops.convert_to_tensor(amsbound, name="amsbound")

    def _apply_dense(self, grad, var):
        exp_avg = self.get_slot(var, "exp_avg")
        exp_avg_sq = self.get_slot(var, "exp_avg_sq")
        if self._amsbound:
            max_exp_avg_sq = self.get_slot(var, "max_exp_avg_sq")

        lr = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1 = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2 = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        final_lr = math_ops.cast(self._final_lr, var.dtype.base_dtype)
        gamma = math_ops.cast(self._gamma, var.dtype.base_dtype)

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported")
