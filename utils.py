import os
import numpy as np
import logging
from logging import info, debug
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from typing import List, Tuple
Tensor = tf.Tensor


def init_placeholders(n_models: int) -> Tuple[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]:
    r"""Create placeholders for all models"""
    inputs = []
    for i in range(n_models):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 10, 1], name=f'x_train_{i}')
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=f'y_train_{i}')
        inputs.append((x, y))
    
    x = tf.placeholder(dtype=tf.float32, shape=[None, 10, 1], name='x_test')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_test')
    test_input = (x, y)
    return inputs, test_input


def init_graph(
    inputs: List[Tuple[Tensor, Tensor]],
    n_models: int,
    optimizers: List[tf.Optimizer],
    model_func,
    split: str = 'train',
):
    r"""Create a parallelized graph of multiple models"""
    list_logits = []
    list_weights = []
    train_ops = []
    losses = []
    for i in range(n_models):
        with tf.variable_scope(f'model_{i}') as scope:
            info(f'adding model {i} to graph')
            x, y = inputs[i]
            logits, weights = model_func(x, split, ...)
            list_logits.append(logits)
            list_weights.append(weights)

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits, name='loss')
            losses.append(loss)
            train_op = optimizers[i].minimize(loss, var_list=weights)
            train_ops.append(train_op)
    return list_logits, list_weights, train_ops, losses


def init_test_graph(
    test_inputs: Tuple[Tensor, Tensor],
    model_func,
    split: str = 'test',
):
    with tf.variable_scope('model_test') as scope:
        x, y = test_inputs
        logits, weights = model_func(x, split, ...)
        y_pred = tf.nn.softmax(logits)
    return y_pred, weights


def import_weights(
    sess: tf.Session,
    list_weights,
    test_weights,
    n_models: int,
):
    r"""Import weights into the graph"""

    assert len(list_weights) == n_models, f"inconsistency: {len(list_weights)} sets of weights and {n_models} models"
    
    for i in range(n_models):
        for w in list_weights[i]:
            pass