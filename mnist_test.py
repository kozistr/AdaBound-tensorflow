from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import os

from tensorflow.examples.tutorials.mnist import input_data

from adabound import AdaBoundOptimizer


def train(sess,
          input_shape=(None, 784), n_classes=10,
          n_feat=32, n_layers=2,
          optimizer="adabound", lr=1e-3, grad_clip=1.,
          log_dir="./logs", model_dir="./model"):
    def prepare_optimizer(optimizer_name="adabound"):
        if optimizer_name == "adabound":
            return AdaBoundOptimizer(learning_rate=lr)
        elif optimizer_name == "adam":
            return tf.train.AdamOptimizer(learning_rate=lr)
        elif optimizer_name == "sgd":
            return tf.train.GradientDescentOptimizer(learning_rate=lr)
        elif optimizer_name == "adagrad":
            return tf.train.AdagradOptimizer(learning_rate=lr)
        elif optimizer_name == "momentum":
            return tf.train.MomentumOptimizer(learning_rate=lr, momentum=1e-6, use_nesterov=True)
        else:
            raise NotImplementedError("[-] Unsupported Optimizer")

    with tf.name_scope("inputs"):
        img = tf.placeholder(tf.float32, shape=input_shape, name="x-image")
        label = tf.placeholder(tf.int32, shape=(None, n_classes), name="y-label")
        do_rate = tf.placeholder(tf.float32, shape=(), name="dropout")

    with tf.variable_scope("simple_cnn_model"):
        x = tf.reshape(img, [-1, 28, 28, 1])

        for n_layer_idx in range(n_layers):
            with tf.variable_scope("cnn_layer_%d" % n_layer_idx):
                x = tf.layers.conv2d(x, filters=n_feat, kernel_size=3, strides=2, padding='SAME')
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = tf.nn.dropout(x, keep_prob=do_rate)

                n_feat *= 2

        x = tf.layers.flatten(x)

        x = tf.layers.dense(x, units=128)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.nn.dropout(x, keep_prob=do_rate)

        logits = tf.layers.dense(x, units=n_classes)
        pred = tf.nn.softmax(logits)

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label))

    with tf.name_scope("train"):
        global_step = tf.train.get_or_create_global_step()
        optimizer = prepare_optimizer(optimizer)

        t_vars = tf.trainable_variables()
        grads = tf.gradients(loss, t_vars)

        # gradient clipping
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=grad_clip)

        train_op = optimizer.apply_gradients(zip(grads, t_vars), global_step=global_step)

        # Normally the global step update is done inside of `apply_gradients`.
        # However, my version of `AdaBoundyOptimizer` doesn't do this. But if you use
        # a different optimizer, you should probably take this line out.
        train_op = tf.group(train_op, [global_step.assign(global_step + 1)])

    with tf.name_scope("metric"):
        corr_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(label, axis=1))
        acc = tf.reduce_mean(tf.cast(corr_pred, dtype=tf.float32))

    with tf.name_scope("summary"):
        tf.summary.scalar("loss/loss", loss)
        tf.summary.scalar("metric/acc", acc)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(log_dir, "train"), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(log_dir, "test"), sess.graph)
    saver = tf.train.Saver(max_to_keep=1)
    return (img, label, do_rate), merged, train_op, loss, (train_writer, test_writer, saver)


def main(training_steps,
         batch_size,
         n_classes,
         learning_rate,
         optimizer,
         filters,
         dropout,
         model_dir,
         data_dir,
         log_dir,
         logging_steps):
    # 0. prepare folders
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. loading the MNIST dataset
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 2. loading the model
        (x, y, do_rate), merged, train_op, loss, (tr_writer, te_writer, saver) = train(
            sess=sess,
            input_shape=(None, 28 * 28),
            n_classes=n_classes,
            n_feat=filters,
            optimizer=optimizer,
            lr=learning_rate,
            log_dir=log_dir
        )

        sess.run(tf.global_variables_initializer())

        # 2-1. loading pre-trained model
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)

            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print("[+] global step : %d" % global_step, " successfully loaded")
        else:
            global_step = 0
            print('[-] No checkpoint file found')

        for steps in range(global_step, training_steps):
            x_tr, y_tr = mnist.train.next_batch(batch_size)
            x_tr /= 255.

            _, loss = sess.run([train_op, loss], feed_dict={
                x: x_tr,
                y: y_tr,
                do_rate: dropout,
            })

            if steps and steps % logging_steps == 0:
                summary = sess.run(merged, feed_dict={
                    x: mnist.test.images / 255.,
                    y: mnist.test.labels,
                    do_rate: 1.,
                })

                te_writer.add_summary(summary, global_step)
                saver.save(sess, model_dir, global_step)

            if steps and steps % logging_steps == 0:
                print("[*] steps %05d : loss %.6f" % (steps, loss))

                summary = sess.run(merged, feed_dict={
                    x: x_tr,
                    y: y_tr,
                    do_rate: dropout,
                })

                tr_writer.add_summary(summary, global_step)

            global_step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_steps', required=False, type=int, default=100000)
    parser.add_argument('--n_classes', required=False, type=int, default=10)
    parser.add_argument('--batch_size', required=False, type=int, default=128)
    parser.add_argument('--learning_rate', required=False, type=float, default=0.001)
    parser.add_argument('--optimizer', required=False, type=str, default="adabound")
    parser.add_argument('--filters', required=False, type=int, default=32)
    parser.add_argument('--dropout', required=False, type=float, default=0.5)
    parser.add_argument('--model_dir', required=False, type=str, default="./model/")
    parser.add_argument('--data_dir', required=False, type=str, default="./mnist/")
    parser.add_argument('--log_dir', required=False, type=str, default="./logs")
    parser.add_argument('--logging_steps', required=False, type=int, default=1000)
    args = vars(parser.parse_args())

    main(
        training_steps=args["training_steps"],
        n_classes=args["n_classes"],
        batch_size=args["batch_size"],
        learning_rate=args["learning_rate"],
        optimizer=args["optimizer"],
        filters=args["filters"],
        dropout=args["dropout"],
        model_dir=args["model_dir"],
        data_dir=args["data_dir"],
        log_dir=args["log_dir"],
        logging_steps=args["logging_steps"],
    )
