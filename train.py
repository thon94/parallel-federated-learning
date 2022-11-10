import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from data import load_data
import logging
from utils import(
    init_placeholders,
    init_graph,
    init_test_graph,
)

def tff_train(
    data_path: str,
    model_func,
    n_models: int,
    lr: float,
    load_ckpt: bool = False,
) -> None:
    train_data, test_data = load_data(data_path)

    train_inputs, test_input = init_placeholders(n_models)

    ## init optimizers for all models
    optimizers = []
    for i in range(n_models):
        optimizers.append(tf.train.AdamOptimizer(learning_rate=lr))

    ## build parallel graph
    list_logits, list_weights, train_ops, losses = init_graph(
        train_inputs,
        n_models,
        optimizers,
        model_func,
        split='train'
    )

    ## build test graph
    y_pred, test_weights = init_test_graph(test_input, model_func, split='test')

    ## clients' weights
    client_weights = {}
    global_weights = None

    ## initialize vars
    init_op = tf.initializers.global_variables()

    ## start training
    with tf.Session() as sess:
        if not load_ckpt:
            rnd = 0
            logging.info('>>> start the federated learning process')
            sess.run(init_op)
            global_weights = list_weights[0]
            client_weights = {i: global_weights for i in range(n_models)}
        else:
            # TODO: load checkpoint to continue training
            # rnd = ???
            logging.info('>>> resume a federated learning process from a checkpoint')
            pass
        
        
        