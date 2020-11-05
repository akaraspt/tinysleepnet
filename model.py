import os
import numpy as np
import sklearn.metrics as skmetrics
import tensorflow as tf
import timeit
import tensorflow.contrib.metrics as contrib_metrics
import tensorflow.contrib.slim as contrib_slim

import nn

import logging
logger = logging.getLogger("default_log")


def create_model(config, use_rnn, output_dir, testing, use_best):
    # Create a model
    if config["model"] == "model-origin":
        model = Model(
            config=config,
            output_dir=output_dir,
            use_rnn=use_rnn,
            testing=testing,
            use_best=use_best
        )
    elif config["model"] == "model-mod-1":
        model = ModelMod1(
            config=config,
            output_dir=output_dir,
            use_rnn=use_rnn,
            testing=testing,
            use_best=use_best
        )
    elif config["model"] == "model-mod-2":
        model = ModelMod2(
            config=config,
            output_dir=output_dir,
            use_rnn=use_rnn,
            testing=testing,
            use_best=use_best
        )
    elif config["model"] == "model-mod-3":
        model = ModelMod3(
            config=config,
            output_dir=output_dir,
            use_rnn=use_rnn,
            testing=testing,
            use_best=use_best
        )
    elif config["model"] == "model-mod-4":
        model = ModelMod4(
            config=config,
            output_dir=output_dir,
            use_rnn=use_rnn,
            testing=testing,
            use_best=use_best
        )
    elif config["model"] == "model-mod-6":
        model = ModelMod6(
            config=config,
            output_dir=output_dir,
            use_rnn=use_rnn,
            testing=testing,
            use_best=use_best
        )
    elif config["model"] == "model-mod-8":
        model = ModelMod8(
            config=config,
            output_dir=output_dir,
            use_rnn=use_rnn,
            testing=testing,
            use_best=use_best
        )
    else:
        raise Exception("Not implemented.")

    return model


class Model(object):

    def __init__(
        self,
        config,
        output_dir="./output",
        use_rnn=False,
        testing=False,
        use_best=False,
    ):
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")
        self.use_rnn = use_rnn

        # Placeholder
        with tf.variable_scope("placeholders") as scope:
            self.signals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1), name='signals')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        # Monitor global step update
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Monitor the number of epochs passed
        self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

        # Build a network that receives inputs from placeholders
        net = self.build_cnn()

        if self.use_rnn:
            # Check whether the corresponding config is given
            if "n_rnn_layers" not in self.config:
                raise Exception("Invalid config.")
            # Append the RNN if needed
            net = self.append_rnn(net)

        # Softmax linear
        net = nn.fc("softmax_linear", net, self.config["n_classes"], bias=0.0)

        # Outputs
        self.logits = net
        self.preds = tf.argmax(self.logits, axis=1)

        # Cross-entropy loss
        self.loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.logits,
            name="loss_ce_per_sample"
        )
        self.loss = tf.reduce_mean(self.loss_per_sample, name="loss_ce_mean")

        # Regularization loss
        reg_losses = self.regularization_loss()

        # Total loss
        self.loss += reg_losses

        # Metrics (used when we want to compute a metric from the output from minibatches)
        with tf.variable_scope("stream_metrics") as scope:
            self.metric_value_op, self.metric_update_op = contrib_metrics.aggregate_metric_map({
                "loss": tf.metrics.mean(values=self.loss),
                "accuracy": tf.metrics.accuracy(labels=self.labels, predictions=self.preds),
                "precision": tf.metrics.precision(labels=self.labels, predictions=self.preds),
                "recall": tf.metrics.recall(labels=self.labels, predictions=self.preds),
            })
            # Manually create reset operations of local vars
            metric_vars = contrib_slim.get_local_variables(scope=scope.name)
            self.metric_init_op = tf.variables_initializer(metric_vars)

        # Training outputs
        self.train_outputs = {
            "global_step": self.global_step,
            "train/loss": self.loss,
            "train/preds": self.preds,
            "train/stream_metrics": self.metric_update_op,
        }
        if self.use_rnn:
            self.train_outputs.update({
                "train/fw_init_state": self.fw_init_state,
                "train/fw_final_state": self.fw_final_state,
                "train/bw_init_state": self.bw_init_state,
                "train/bw_final_state": self.bw_final_state,
            })

        # Test outputs
        self.test_outputs = {
            "global_step": self.global_step,
            "test/loss": self.loss,
            "test/preds": self.preds,
        }
        if self.use_rnn:
            self.test_outputs.update({
                "test/fw_init_state": self.fw_init_state,
                "test/fw_final_state": self.fw_final_state,
                "test/bw_init_state": self.bw_init_state,
                "test/bw_final_state": self.bw_final_state,
            })

        # Tensoflow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        if not testing:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"))
            self.train_writer.add_graph(self.sess.graph)
            logger.info("Saved tensorboard graph to {}".format(self.train_writer.get_logdir()))

        # Optimizer
        if not testing:
            with tf.variable_scope("optimizer") as scope:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Pretraining
                    if not self.use_rnn:
                        self.train_step_op, self.grad_op = nn.adam_optimizer(
                            loss=self.loss,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            learning_rate=self.config["learning_rate"],
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                        )
                    # Fine-tuning
                    else:
                        # Use different learning rates for CNN and RNN
                        cnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cnn")
                        remainin_vars = list(set(tf.trainable_variables()) - set(cnn_vars))
                        self.train_step_op, self.grad_op = nn.adam_optimizer_clip_lrs(
                            loss=self.loss,
                            list_train_vars=[cnn_vars, remainin_vars],
                            list_lrs=[self.config["lr_cnn"], self.config["lr_rnn"]],
                            global_step=self.global_step,
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                            clip_value=self.config["clip_grad_value"],
                        )

        # Initializer
        with tf.variable_scope("initializer") as scope:
            # tf.trainable_variables() or tf.global_variables()
            self.init_global_op = tf.variables_initializer(tf.global_variables())
            self.init_local_op = tf.variables_initializer(tf.local_variables())

        # Saver for storing variables
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Initialize variables
        self.run([self.init_global_op, self.init_local_op])

        # Restore variables (if possible)
        is_restore = False
        if use_best:
            if os.path.exists(self.best_ckpt_path):
                if os.path.isfile(os.path.join(self.best_ckpt_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.best_ckpt_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    logger.info("Best model restored from {}".format(latest_checkpoint))
                    is_restore = True
        else:
            if os.path.exists(self.checkpoint_path):
                if os.path.isfile(os.path.join(self.checkpoint_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    logger.info("Model restored from {}".format(latest_checkpoint))
                    is_restore = True
        if not is_restore:
            logger.info("Model started from random weights")

    def get_current_epoch(self):
        return self.run(self.global_epoch)

    def pass_one_epoch(self):
        self.run(tf.assign(self.global_epoch, self.global_epoch+1))

    def run(self, *args, **kwargs):
        return self.sess.run(*args, **kwargs)

    def save_checkpoint(self, name):
        path = self.saver.save(
            self.sess,
            os.path.join(self.checkpoint_path, "{}.ckpt".format(name)),
            global_step=self.global_step
        )
        logger.info("Saved checkpoint to {}".format(path))

    def save_best_checkpoint(self, name):
        path = self.best_saver.save(
            self.sess,
            os.path.join(self.best_ckpt_path, "{}.ckpt".format(name)),
            global_step=self.global_step
        )
        logger.info("Saved best checkpoint to {}".format(path))

    def save_weights(self, scope, name, key_variables=tf.GraphKeys.TRAINABLE_VARIABLES):
        # Save weights
        path = os.path.join(self.weights_path, "{}.npz".format(name))
        logger.info("Saving weights in scope: {} to {}".format(scope, path))
        save_dict = {}
        cnn_vars = tf.get_collection(key_variables, scope=scope)
        for v in cnn_vars:
            save_dict[v.name] = self.sess.run(v)
            logger.info("  variable: {}".format(v.name))
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        np.savez(path, **save_dict)

    def load_weights(self, scope, weight_file, key_variables=tf.GraphKeys.TRAINABLE_VARIABLES):
        # Load weights
        logger.info("Loading weights in scope: {} from {}".format(scope, weight_file))
        cnn_vars = tf.get_collection(key_variables, scope=scope)
        with np.load(weight_file) as f:
            for v in cnn_vars:
                tensor = tf.get_default_graph().get_tensor_by_name(v.name)
                self.run(tf.assign(tensor, f[v.name]))
                logger.info("  variable: {}".format(v.name))

    def regularization_loss(self):
        reg_losses = []
        for v in tf.trainable_variables():
            if "cnn_small/conv1d_1/conv2d/kernel:0" in v.name:
                reg_losses.append(tf.nn.l2_loss(v))
        reg_losses = tf.multiply(tf.add_n(reg_losses, name="l2_loss"), self.config["l2_weight_decay"])
        return reg_losses

    def build_cnn(self):
        cnn_small_size = int(self.config["sampling_rate"] / 2.0)
        cnn_large_size = int(self.config["sampling_rate"] * 4.0)
        cnn_small_stride = int(self.config["sampling_rate"] / 16.0)
        cnn_large_stride = int(self.config["sampling_rate"] / 2.0)

        cnn_outputs = []
        with tf.variable_scope("cnn") as scope:
            # CNN-small
            with tf.variable_scope("cnn_small") as scope:
                net = nn.conv1d("conv1d_1", self.signals, 64, cnn_small_size, cnn_small_stride)
                net = nn.batch_norm("bn_1", net, self.is_training)
                net = tf.nn.relu(net, name="relu_1")

                net = nn.max_pool1d("maxpool1d_1", net, 8, 8)

                net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_1")

                net = nn.conv1d("conv1d_2_1", net, 128, 8, 1)
                net = nn.batch_norm("bn_2_1", net, self.is_training)
                net = tf.nn.relu(net, name="relu_2_1")
                net = nn.conv1d("conv1d_2_2", net, 128, 8, 1)
                net = nn.batch_norm("bn_2_2", net, self.is_training)
                net = tf.nn.relu(net, name="relu_2_2")
                net = nn.conv1d("conv1d_2_3", net, 128, 8, 1)
                net = nn.batch_norm("bn_2_3", net, self.is_training)
                net = tf.nn.relu(net, name="relu_2_3")

                net = nn.max_pool1d("maxpool1d_2", net, 4, 4)

                net = tf.layers.flatten(net, name="flatten_2")

                cnn_outputs.append(net)

            # CNN-large
            with tf.variable_scope("cnn_large") as scope:
                net = nn.conv1d("conv1d_1", self.signals, 64, cnn_large_size, cnn_large_stride)
                net = nn.batch_norm("bn_1", net, self.is_training)
                net = tf.nn.relu(net, name="relu_1")

                net = nn.max_pool1d("maxpool1d_1", net, 4, 4)

                net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_1")

                net = nn.conv1d("conv1d_2_1", net, 128, 6, 1)
                net = nn.batch_norm("bn_2_1", net, self.is_training)
                net = tf.nn.relu(net, name="relu_2_1")
                net = nn.conv1d("conv1d_2_2", net, 128, 6, 1)
                net = nn.batch_norm("bn_2_2", net, self.is_training)
                net = tf.nn.relu(net, name="relu_2_2")
                net = nn.conv1d("conv1d_2_3", net, 128, 6, 1)
                net = nn.batch_norm("bn_2_3", net, self.is_training)
                net = tf.nn.relu(net, name="relu_2_3")

                net = nn.max_pool1d("maxpool1d_2", net, 2, 2)

                net = tf.layers.flatten(net, name="flatten_2")

                cnn_outputs.append(net)

        # Merge output from two CNNs
        net = tf.concat(values=cnn_outputs, axis=1, name="concat_3")

        net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_3")

        return net

    def append_rnn(self, inputs):
        output_conns = []

        # RNN
        with tf.variable_scope("rnn") as scope:
            # Fully-connected for shortcut connection
            with tf.variable_scope("shortcut") as scope:
                shortcut = nn.fc("fc", inputs, self.config["n_rnn_units"]*2)
                shortcut = nn.batch_norm("bn", shortcut, self.is_training)
                shortcut = tf.nn.relu(shortcut, name="relu")
                output_conns.append(shortcut)

            # Reshape the input from (batch_size * seq_length, input_dim) to
            # (batch_size, seq_length, input_dim)
            input_dim = inputs.shape[-1].value
            seq_inputs = tf.reshape(inputs, shape=[-1, self.config["seq_length"], input_dim], name="reshape_seq_inputs")

            def _create_rnn_cell(n_units):
                """A function to create a new rnn cell."""
                cell = tf.contrib.rnn.LSTMCell(
                    num_units=n_units,
                    use_peepholes=True,
                    forget_bias=1.0,
                    state_is_tuple=True,
                )
                # Dropout wrapper
                keep_prob = tf.cond(self.is_training, lambda:tf.constant(0.5), lambda:tf.constant(1.0))
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
                return cell

            # Create forward and backward RNN cells
            fw_cells = []
            bw_cells = []
            for l in range(self.config["n_rnn_layers"]):
                fw_cells.append(_create_rnn_cell(self.config["n_rnn_units"]))
                bw_cells.append(_create_rnn_cell(self.config["n_rnn_units"]))

            # Multiple layers of forward and backward cells
            multi_fw_cell = tf.nn.rnn_cell.MultiRNNCell(cells=fw_cells, state_is_tuple=True)
            multi_bw_cell = tf.nn.rnn_cell.MultiRNNCell(cells=bw_cells, state_is_tuple=True)

            # Initial states
            self.fw_init_state = multi_fw_cell.zero_state(self.config["batch_size"], tf.float32)
            self.bw_init_state = multi_bw_cell.zero_state(self.config["batch_size"], tf.float32)

            # Create bidirectional rnn
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=multi_fw_cell,
                cell_bw=multi_bw_cell,
                inputs=seq_inputs,
                # sequence_length=[self.config["rnn_seq_length"]]*self.config["batch_size_rnn"],
                initial_state_fw=self.fw_init_state,
                initial_state_bw=self.bw_init_state,
            )
            # Final states
            self.fw_final_state, self.bw_final_state = states

            # Concatenate the output from forward and backward cells
            net = tf.concat(values=outputs, axis=-1, name="merge_fw_bw_output")
            net = tf.reshape(net, shape=[-1, self.config["n_rnn_units"]*2], name="reshape_nonseq_input")

            output_conns.append(net)

        # Element-wise add outputs of shortcut and bidirect_rnn (residual technique)
        net = tf.add_n(output_conns, name="add")

        net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop")

        return net

    def train(self, minibatches):
        self.run(self.metric_init_op)
        start = timeit.default_timer()
        preds = []
        trues = []

        if self.use_rnn:
            # Initialize state of RNN - Bidirectional
            fw_state = self.run(self.fw_init_state)
            bw_state = self.run(self.bw_init_state)

        for x, y in minibatches:
            feed_dict = {
                self.signals: x,
                self.labels: y,
                self.is_training: True,
            }

            if self.use_rnn:
                # Carry the states from the previous batches through time
                for i, (c, h) in enumerate(self.fw_init_state):
                    feed_dict[c] = fw_state[i].c
                    feed_dict[h] = fw_state[i].h
                for i, (c, h) in enumerate(self.bw_init_state):
                    feed_dict[c] = bw_state[i].c
                    feed_dict[h] = bw_state[i].h

            _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)

            if self.use_rnn:
                # Buffer the final states
                fw_state = outputs["train/fw_final_state"]
                bw_state = outputs["train/bw_final_state"]

            preds.extend(outputs["train/preds"])
            trues.extend(y)

        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs.update({
            "train/trues": trues,
            "train/preds": preds,
            "train/accuracy": acc,
            "train/f1_score": f1_score,
            "train/cm": cm,
            "train/duration": duration,
        })
        return outputs

    def evaluate(self, minibatches):
        start = timeit.default_timer()
        losses = []
        preds = []
        trues = []

        if self.use_rnn:
            # Initialize state of RNN - Bidirectional
            fw_state = self.run(self.fw_init_state)
            bw_state = self.run(self.bw_init_state)

        for x, y in minibatches:
            feed_dict = {
                self.signals: x,
                self.labels: y,
                self.is_training: False,
            }

            if self.use_rnn:
                # Carry the states from the previous batches through time
                for i, (c, h) in enumerate(self.fw_init_state):
                    feed_dict[c] = fw_state[i].c
                    feed_dict[h] = fw_state[i].h
                for i, (c, h) in enumerate(self.bw_init_state):
                    feed_dict[c] = bw_state[i].c
                    feed_dict[h] = bw_state[i].h

            outputs = self.run(self.test_outputs, feed_dict=feed_dict)

            if self.use_rnn:
                # Buffer the final states
                fw_state = outputs["test/fw_final_state"]
                bw_state = outputs["test/bw_final_state"]

            losses.append(outputs["test/loss"])
            preds.extend(outputs["test/preds"])
            trues.extend(y)

        loss = np.mean(losses)
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs = {
            "test/trues": trues,
            "test/preds": preds,
            "test/loss": loss,
            "test/accuracy": acc,
            "test/f1_score": f1_score,
            "test/cm": cm,
            "test/duration": duration,
        }
        return outputs


class ModelMod1(Model):
    '''Change from bi-LSTM to LSTM.'''

    def __init__(
        self,
        config,
        output_dir="./output",
        use_rnn=False,
        testing=False,
        use_best=False,
    ):
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")
        self.use_rnn = use_rnn

        # Placeholder
        with tf.variable_scope("placeholders") as scope:
            self.signals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1), name='signals')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

            if self.use_rnn:
                self.loss_weights = tf.placeholder(dtype=tf.float32, shape=(None, ), name='loss_weights')
                self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=(None, ), name='seq_lengths')

        # Monitor global step update
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Monitor the number of epochs passed
        self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

        # Build a network that receives inputs from placeholders
        net = self.build_cnn()

        if self.use_rnn:
            # Check whether the corresponding config is given
            if "n_rnn_layers" not in self.config:
                raise Exception("Invalid config.")
            # Append the RNN if needed
            net = self.append_rnn(net)

        # Softmax linear
        net = nn.fc("softmax_linear", net, self.config["n_classes"], bias=0.0)

        # Outputs
        self.logits = net
        self.preds = tf.argmax(self.logits, axis=1)

        # Cross-entropy loss
        self.loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.logits,
            name="loss_ce_per_sample"
        )

        with tf.name_scope("loss_ce_mean") as scope:
            if self.use_rnn:
                self.loss = tf.multiply(self.loss_weights, self.loss_per_sample)
                self.loss = tf.reduce_sum(self.loss) / tf.reduce_sum(self.loss_weights)
            else:
                self.loss = tf.reduce_mean(self.loss_per_sample)

        # Regularization loss
        reg_losses = self.regularization_loss()

        # Total loss
        self.loss += reg_losses

        # Metrics (used when we want to compute a metric from the output from minibatches)
        with tf.variable_scope("stream_metrics") as scope:
            self.metric_value_op, self.metric_update_op = contrib_metrics.aggregate_metric_map({
                "loss": tf.metrics.mean(values=self.loss),
                "accuracy": tf.metrics.accuracy(labels=self.labels, predictions=self.preds),
                "precision": tf.metrics.precision(labels=self.labels, predictions=self.preds),
                "recall": tf.metrics.recall(labels=self.labels, predictions=self.preds),
            })
            # Manually create reset operations of local vars
            metric_vars = contrib_slim.get_local_variables(scope=scope.name)
            self.metric_init_op = tf.variables_initializer(metric_vars)

        # Training outputs
        self.train_outputs = {
            "global_step": self.global_step,
            "train/loss": self.loss,
            "train/preds": self.preds,
            "train/stream_metrics": self.metric_update_op,
        }
        if self.use_rnn:
            self.train_outputs.update({
                "train/init_state": self.init_state,
                "train/final_state": self.final_state,
            })

        # Test outputs
        self.test_outputs = {
            "global_step": self.global_step,
            "test/loss": self.loss,
            "test/preds": self.preds,
        }
        if self.use_rnn:
            self.test_outputs.update({
                "test/init_state": self.init_state,
                "test/final_state": self.final_state,
            })

        # Tensoflow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        if not testing:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"))
            self.train_writer.add_graph(self.sess.graph)
            logger.info("Saved tensorboard graph to {}".format(self.train_writer.get_logdir()))

        # Optimizer
        if not testing:
            with tf.variable_scope("optimizer") as scope:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Pretraining
                    if not self.use_rnn:
                        self.train_step_op, self.grad_op = nn.adam_optimizer(
                            loss=self.loss,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            learning_rate=self.config["learning_rate"],
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                        )
                    # Fine-tuning
                    else:
                        # Use different learning rates for CNN and RNN
                        cnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cnn")
                        remainin_vars = list(set(tf.trainable_variables()) - set(cnn_vars))
                        self.train_step_op, self.grad_op = nn.adam_optimizer_clip_lrs(
                            loss=self.loss,
                            list_train_vars=[cnn_vars, remainin_vars],
                            list_lrs=[self.config["lr_cnn"], self.config["lr_rnn"]],
                            global_step=self.global_step,
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                            clip_value=self.config["clip_grad_value"],
                        )

        # Initializer
        with tf.variable_scope("initializer") as scope:
            # tf.trainable_variables() or tf.global_variables()
            self.init_global_op = tf.variables_initializer(tf.global_variables())
            self.init_local_op = tf.variables_initializer(tf.local_variables())

        # Saver for storing variables
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Initialize variables
        self.run([self.init_global_op, self.init_local_op])

        # Restore variables (if possible)
        is_restore = False
        if use_best:
            if os.path.exists(self.best_ckpt_path):
                if os.path.isfile(os.path.join(self.best_ckpt_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.best_ckpt_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    logger.info("Best model restored from {}".format(latest_checkpoint))
                    is_restore = True
        else:
            if os.path.exists(self.checkpoint_path):
                if os.path.isfile(os.path.join(self.checkpoint_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    logger.info("Model restored from {}".format(latest_checkpoint))
                    is_restore = True
        if not is_restore:
            logger.info("Model started from random weights")

    def append_rnn(self, inputs):
        output_conns = []

        # RNN
        with tf.variable_scope("rnn") as scope:
            # Fully-connected for shortcut connection
            with tf.variable_scope("shortcut") as scope:
                shortcut = nn.fc("fc", inputs, self.config["n_rnn_units"])
                shortcut = nn.batch_norm("bn", shortcut, self.is_training)
                shortcut = tf.nn.relu(shortcut, name="relu")
                output_conns.append(shortcut)

            # Reshape the input from (batch_size * seq_length, input_dim) to
            # (batch_size, seq_length, input_dim)
            input_dim = inputs.shape[-1].value
            seq_inputs = tf.reshape(inputs, shape=[-1, self.config["seq_length"], input_dim], name="reshape_seq_inputs")

            def _create_rnn_cell(n_units):
                """A function to create a new rnn cell."""
                cell = tf.contrib.rnn.LSTMCell(
                    num_units=n_units,
                    use_peepholes=True,
                    forget_bias=1.0,
                    state_is_tuple=True,
                )
                # Dropout wrapper
                keep_prob = tf.cond(self.is_training, lambda:tf.constant(0.5), lambda:tf.constant(1.0))
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
                return cell

            # LSTM
            cells = []
            for l in range(self.config["n_rnn_layers"]):
                cells.append(_create_rnn_cell(self.config["n_rnn_units"]))

            # Multiple layers of forward and backward cells
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)

            # Initial states
            self.init_state = multi_cell.zero_state(self.config["batch_size"], tf.float32)

            # Create rnn
            outputs, states = tf.nn.dynamic_rnn(
                cell=multi_cell,
                inputs=seq_inputs,
                initial_state=self.init_state,
                sequence_length=self.seq_lengths,
            )

            # Final states
            self.final_state = states

            # Concatenate the output from forward and backward cells
            net = tf.reshape(outputs, shape=[-1, self.config["n_rnn_units"]], name="reshape_nonseq_input")

            output_conns.append(net)

        # Element-wise add outputs of shortcut and bidirect_rnn (residual technique)
        net = tf.add_n(output_conns, name="add")

        # net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop")

        return net

    def train(self, minibatches):
        self.run(self.metric_init_op)
        start = timeit.default_timer()
        preds = []
        trues = []

        if not self.use_rnn:
            for x, y in minibatches:
                feed_dict = {
                    self.signals: x,
                    self.labels: y,
                    self.is_training: True,
                }

                _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)

                preds.extend(outputs["train/preds"])
                trues.extend(y)
        else:
            for x, y, w, sl, re in minibatches:
                feed_dict = {
                    self.signals: x,
                    self.labels: y,
                    self.is_training: True,
                    self.loss_weights: w,
                    self.seq_lengths: sl,
                }

                if re:
                    # Initialize state of RNN
                    state = self.run(self.init_state)

                # Carry the states from the previous batches through time
                for i, (c, h) in enumerate(self.init_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

                _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)

                # Buffer the final states
                state = outputs["train/final_state"]

                tmp_preds = np.reshape(outputs["train/preds"], (self.config["batch_size"], self.config["seq_length"]))
                tmp_trues = np.reshape(y, (self.config["batch_size"], self.config["seq_length"]))

                for i in range(self.config["batch_size"]):
                    preds.extend(tmp_preds[i, :sl[i]])
                    trues.extend(tmp_trues[i, :sl[i]])

        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs.update({
            "train/trues": trues,
            "train/preds": preds,
            "train/accuracy": acc,
            "train/f1_score": f1_score,
            "train/cm": cm,
            "train/duration": duration,
        })
        return outputs

    def evaluate(self, minibatches):
        start = timeit.default_timer()
        losses = []
        preds = []
        trues = []

        if not self.use_rnn:
            for x, y in minibatches:
                feed_dict = {
                    self.signals: x,
                    self.labels: y,
                    self.is_training: False,
                }

                outputs = self.run(self.test_outputs, feed_dict=feed_dict)

                losses.append(outputs["test/loss"])
                preds.extend(outputs["test/preds"])
                trues.extend(y)
        else:
            for x, y, w, sl, re in minibatches:
                feed_dict = {
                    self.signals: x,
                    self.labels: y,
                    self.is_training: False,
                    self.loss_weights: w,
                    self.seq_lengths: sl,
                }

                if re:
                    # Initialize state of RNN
                    state = self.run(self.init_state)

                # Carry the states from the previous batches through time
                for i, (c, h) in enumerate(self.init_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

                outputs = self.run(self.test_outputs, feed_dict=feed_dict)

                # Buffer the final states
                state = outputs["test/final_state"]

                losses.append(outputs["test/loss"])

                tmp_preds = np.reshape(outputs["test/preds"], (self.config["batch_size"], self.config["seq_length"]))
                tmp_trues = np.reshape(y, (self.config["batch_size"], self.config["seq_length"]))

                for i in range(self.config["batch_size"]):
                    preds.extend(tmp_preds[i, :sl[i]])
                    trues.extend(tmp_trues[i, :sl[i]])

        loss = np.mean(losses)
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs = {
            "test/trues": trues,
            "test/preds": preds,
            "test/loss": loss,
            "test/accuracy": acc,
            "test/f1_score": f1_score,
            "test/cm": cm,
            "test/duration": duration,
        }
        return outputs


class ModelMod2(ModelMod1):
    '''Reduce from two CNNs to one CNNs.'''

    def regularization_loss(self):
        reg_losses = []
        list_vars = [
            # "cnn/conv1d_1/conv2d/kernel:0",
            # "cnn/conv1d_2_1/conv2d/kernel:0",
            # "cnn/conv1d_2_2/conv2d/kernel:0",
            # "cnn/conv1d_2_3/conv2d/kernel:0",
            # "rnn/shortcut/fc/dense/kernel:0",
            # "softmax_linear/dense/kernel:0",
        ]
        for v in tf.trainable_variables():
            if any(v.name in s for s in list_vars):
                reg_losses.append(tf.nn.l2_loss(v))
        if len(reg_losses):
            reg_losses = tf.multiply(tf.add_n(reg_losses, name="l2_loss"), self.config["l2_weight_decay"])
        else:
            reg_losses = 0
        return reg_losses

    def build_cnn(self):
        first_filter_size = int(self.config["sampling_rate"] / 2.0)
        first_filter_stride = int(self.config["sampling_rate"] / 16.0)

        with tf.variable_scope("cnn") as scope:
            net = nn.conv1d("conv1d_1", self.signals, 128, first_filter_size, first_filter_stride)
            net = nn.batch_norm("bn_1", net, self.is_training)
            net = tf.nn.relu(net, name="relu_1")

            net = nn.max_pool1d("maxpool1d_1", net, 8, 8)

            net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_1")

            net = nn.conv1d("conv1d_2_1", net, 128, 8, 1)
            net = nn.batch_norm("bn_2_1", net, self.is_training)
            net = tf.nn.relu(net, name="relu_2_1")
            net = nn.conv1d("conv1d_2_2", net, 128, 8, 1)
            net = nn.batch_norm("bn_2_2", net, self.is_training)
            net = tf.nn.relu(net, name="relu_2_2")
            net = nn.conv1d("conv1d_2_3", net, 128, 8, 1)
            net = nn.batch_norm("bn_2_3", net, self.is_training)
            net = tf.nn.relu(net, name="relu_2_3")

            net = nn.max_pool1d("maxpool1d_2", net, 4, 4)

            net = tf.layers.flatten(net, name="flatten_2")

        net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_2")

        return net


class ModelMod3(ModelMod2):
    '''Remove shortcut connection.'''

    def __init__(
        self,
        config,
        output_dir="./output",
        use_rnn=False,
        testing=False,
        use_best=False,
    ):
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")
        self.use_rnn = use_rnn

        # Placeholder
        with tf.variable_scope("placeholders") as scope:
            self.signals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1), name='signals')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

            if self.use_rnn:
                self.loss_weights = tf.placeholder(dtype=tf.float32, shape=(None, ), name='loss_weights')
                self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=(None, ), name='seq_lengths')

        # Monitor global step update
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Monitor the number of epochs passed
        self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

        # Build a network that receives inputs from placeholders
        net = self.build_cnn()

        if self.use_rnn:
            # Check whether the corresponding config is given
            if "n_rnn_layers" not in self.config:
                raise Exception("Invalid config.")
            # Append the RNN if needed
            net = self.append_rnn(net)

        # Softmax linear
        net = nn.fc("softmax_linear", net, self.config["n_classes"], bias=0.0)

        # Outputs
        self.logits = net
        self.preds = tf.argmax(self.logits, axis=1)

        # Cross-entropy loss
        self.loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.logits,
            name="loss_ce_per_sample"
        )

        with tf.name_scope("loss_ce_mean") as scope:
            if self.use_rnn:
                self.loss = tf.multiply(self.loss_weights, self.loss_per_sample)
                self.loss = tf.reduce_sum(self.loss) / tf.reduce_sum(self.loss_weights)
            else:
                self.loss = tf.reduce_mean(self.loss_per_sample)

        # Regularization loss
        reg_losses = self.regularization_loss()

        # Total loss
        self.loss += reg_losses

        # Metrics (used when we want to compute a metric from the output from minibatches)
        with tf.variable_scope("stream_metrics") as scope:
            self.metric_value_op, self.metric_update_op = contrib_metrics.aggregate_metric_map({
                "loss": tf.metrics.mean(values=self.loss),
                "accuracy": tf.metrics.accuracy(labels=self.labels, predictions=self.preds),
                "precision": tf.metrics.precision(labels=self.labels, predictions=self.preds),
                "recall": tf.metrics.recall(labels=self.labels, predictions=self.preds),
            })
            # Manually create reset operations of local vars
            metric_vars = contrib_slim.get_local_variables(scope=scope.name)
            self.metric_init_op = tf.variables_initializer(metric_vars)

        # Training outputs
        self.train_outputs = {
            "global_step": self.global_step,
            "train/loss": self.loss,
            "train/preds": self.preds,
            "train/stream_metrics": self.metric_update_op,
        }
        if self.use_rnn:
            self.train_outputs.update({
                "train/init_state": self.init_state,
                "train/final_state": self.final_state,
            })

        # Test outputs
        self.test_outputs = {
            "global_step": self.global_step,
            "test/loss": self.loss,
            "test/preds": self.preds,
        }
        if self.use_rnn:
            self.test_outputs.update({
                "test/init_state": self.init_state,
                "test/final_state": self.final_state,
            })

        # Tensoflow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        if not testing:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"))
            self.train_writer.add_graph(self.sess.graph)
            logger.info("Saved tensorboard graph to {}".format(self.train_writer.get_logdir()))

        # Optimizer
        if not testing:
            with tf.variable_scope("optimizer") as scope:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Pretraining
                    if not self.use_rnn:
                        self.train_step_op, self.grad_op = nn.adam_optimizer(
                            loss=self.loss,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            learning_rate=self.config["learning_rate"],
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                        )
                    # Fine-tuning
                    else:
                        # Use different learning rates for CNN and RNN
                        self.train_step_op, self.grad_op = nn.adam_optimizer_clip(
                            loss=self.loss,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            learning_rate=self.config["learning_rate"],
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                            clip_value=self.config["clip_grad_value"],
                        )

        # Initializer
        with tf.variable_scope("initializer") as scope:
            # tf.trainable_variables() or tf.global_variables()
            self.init_global_op = tf.variables_initializer(tf.global_variables())
            self.init_local_op = tf.variables_initializer(tf.local_variables())

        # Saver for storing variables
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Initialize variables
        self.run([self.init_global_op, self.init_local_op])

        # Restore variables (if possible)
        is_restore = False
        if use_best:
            if os.path.exists(self.best_ckpt_path):
                if os.path.isfile(os.path.join(self.best_ckpt_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.best_ckpt_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    logger.info("Best model restored from {}".format(latest_checkpoint))
                    is_restore = True
        else:
            if os.path.exists(self.checkpoint_path):
                if os.path.isfile(os.path.join(self.checkpoint_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    logger.info("Model restored from {}".format(latest_checkpoint))
                    is_restore = True
        if not is_restore:
            logger.info("Model started from random weights")

    def append_rnn(self, inputs):
        # RNN
        with tf.variable_scope("rnn") as scope:
            # Reshape the input from (batch_size * seq_length, input_dim) to
            # (batch_size, seq_length, input_dim)
            input_dim = inputs.shape[-1].value
            seq_inputs = tf.reshape(inputs, shape=[-1, self.config["seq_length"], input_dim], name="reshape_seq_inputs")

            def _create_rnn_cell(n_units):
                """A function to create a new rnn cell."""
                cell = tf.contrib.rnn.LSTMCell(
                    num_units=n_units,
                    use_peepholes=True,
                    forget_bias=1.0,
                    state_is_tuple=True,
                )
                # Dropout wrapper
                keep_prob = tf.cond(self.is_training, lambda:tf.constant(0.5), lambda:tf.constant(1.0))
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
                return cell

            # LSTM
            cells = []
            for l in range(self.config["n_rnn_layers"]):
                cells.append(_create_rnn_cell(self.config["n_rnn_units"]))

            # Multiple layers of forward and backward cells
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)

            # Initial states
            self.init_state = multi_cell.zero_state(self.config["batch_size"], tf.float32)

            # Create rnn
            outputs, states = tf.nn.dynamic_rnn(
                cell=multi_cell,
                inputs=seq_inputs,
                initial_state=self.init_state,
                sequence_length=self.seq_lengths,
            )

            # Final states
            self.final_state = states

            # Concatenate the output from forward and backward cells
            net = tf.reshape(outputs, shape=[-1, self.config["n_rnn_units"]], name="reshape_nonseq_input")

            net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop")

        return net

    def regularization_loss(self):
        reg_losses = []
        list_vars = [
            "cnn/conv1d_1/conv2d/kernel:0",
            "cnn/conv1d_2_1/conv2d/kernel:0",
            "cnn/conv1d_2_2/conv2d/kernel:0",
            "cnn/conv1d_2_3/conv2d/kernel:0",
            # "rnn/shortcut/fc/dense/kernel:0",
            # "softmax_linear/dense/kernel:0",
        ]
        for v in tf.trainable_variables():
            if any(v.name in s for s in list_vars):
                reg_losses.append(tf.nn.l2_loss(v))
        if len(reg_losses):
            reg_losses = tf.multiply(tf.add_n(reg_losses, name="l2_loss"), self.config["l2_weight_decay"])
        else:
            reg_losses = 0
        return reg_losses


class ModelMod4(ModelMod3):
    '''Add weighted loss.'''

    def __init__(
        self,
        config,
        output_dir="./output",
        use_rnn=False,
        testing=False,
        use_best=False,
    ):
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")
        self.use_rnn = use_rnn

        # Placeholder
        with tf.variable_scope("placeholders") as scope:
            self.signals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1), name='signals')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

            if self.use_rnn:
                self.loss_weights = tf.placeholder(dtype=tf.float32, shape=(None, ), name='loss_weights')
                self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=(None, ), name='seq_lengths')

        # Monitor global step update
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Monitor the number of epochs passed
        self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

        # Build a network that receives inputs from placeholders
        net = self.build_cnn()

        if self.use_rnn:
            # Check whether the corresponding config is given
            if "n_rnn_layers" not in self.config:
                raise Exception("Invalid config.")
            # Append the RNN if needed
            net = self.append_rnn(net)

        # Softmax linear
        net = nn.fc("softmax_linear", net, self.config["n_classes"], bias=0.0)

        # Outputs
        self.logits = net
        self.preds = tf.argmax(self.logits, axis=1)

        # Cross-entropy loss
        self.loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.logits,
            name="loss_ce_per_sample"
        )

        with tf.name_scope("loss_ce_mean") as scope:
            if self.use_rnn:
                # Weight by sequence
                loss_w_seq = tf.multiply(self.loss_weights, self.loss_per_sample)

                # Weight by class
                sample_weights = tf.reduce_sum(
                    tf.multiply(
                        tf.one_hot(indices=self.labels, depth=self.config["n_classes"]), 
                        np.asarray(self.config["class_weights"], dtype=np.float32)
                    ), 1
                )
                loss_w_class = tf.multiply(loss_w_seq, sample_weights)

                # Computer average loss scaled with the sequence length
                self.loss_ce = tf.reduce_sum(loss_w_class) / tf.reduce_sum(self.loss_weights)
            else:
                self.loss_ce = tf.reduce_mean(self.loss_per_sample)

        # Regularization loss
        self.reg_losses = self.regularization_loss()

        # Total loss
        self.loss = self.loss_ce + self.reg_losses

        # Metrics (used when we want to compute a metric from the output from minibatches)
        with tf.variable_scope("stream_metrics") as scope:
            self.metric_value_op, self.metric_update_op = contrib_metrics.aggregate_metric_map({
                "loss": tf.metrics.mean(values=self.loss),
                "accuracy": tf.metrics.accuracy(labels=self.labels, predictions=self.preds),
                "precision": tf.metrics.precision(labels=self.labels, predictions=self.preds),
                "recall": tf.metrics.recall(labels=self.labels, predictions=self.preds),
            })
            # Manually create reset operations of local vars
            metric_vars = contrib_slim.get_local_variables(scope=scope.name)
            self.metric_init_op = tf.variables_initializer(metric_vars)

        # Training outputs
        self.train_outputs = {
            "global_step": self.global_step,
            "train/loss": self.loss,
            "train/preds": self.preds,
            "train/stream_metrics": self.metric_update_op,
        }
        if self.use_rnn:
            self.train_outputs.update({
                "train/init_state": self.init_state,
                "train/final_state": self.final_state,
            })

        # Test outputs
        self.test_outputs = {
            "global_step": self.global_step,
            "test/loss": self.loss,
            "test/preds": self.preds,
        }
        if self.use_rnn:
            self.test_outputs.update({
                "test/init_state": self.init_state,
                "test/final_state": self.final_state,
            })

        # Tensoflow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        if not testing:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"))
            self.train_writer.add_graph(self.sess.graph)
            logger.info("Saved tensorboard graph to {}".format(self.train_writer.get_logdir()))

        # Optimizer
        if not testing:
            # self.lr = tf.train.exponential_decay(
            #     learning_rate=self.config["learning_rate_decay"],
            #     global_step=self.global_step,
            #     decay_steps=self.config["decay_steps"],
            #     decay_rate=self.config["decay_rate"],
            #     staircase=False,
            #     name="learning_rate"
            # )
            self.lr = tf.constant(self.config["learning_rate"], dtype=tf.float32)
            with tf.variable_scope("optimizer") as scope:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Pretraining
                    if not self.use_rnn:
                        self.train_step_op, self.grad_op = nn.adam_optimizer(
                            loss=self.loss,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            # learning_rate=self.config["learning_rate"],
                            learning_rate=self.lr,
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                        )
                    # Fine-tuning
                    else:
                        # Use different learning rates for CNN and RNN
                        self.train_step_op, self.grad_op = nn.adam_optimizer_clip(
                            loss=self.loss,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            # learning_rate=self.config["learning_rate"],
                            learning_rate=self.lr,
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                            clip_value=self.config["clip_grad_value"],
                        )

        # Initializer
        with tf.variable_scope("initializer") as scope:
            # tf.trainable_variables() or tf.global_variables()
            self.init_global_op = tf.variables_initializer(tf.global_variables())
            self.init_local_op = tf.variables_initializer(tf.local_variables())

        # Saver for storing variables
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Initialize variables
        self.run([self.init_global_op, self.init_local_op])

        # Restore variables (if possible)
        is_restore = False
        if use_best:
            if os.path.exists(self.best_ckpt_path):
                if os.path.isfile(os.path.join(self.best_ckpt_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.best_ckpt_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    logger.info("Best model restored from {}".format(latest_checkpoint))
                    is_restore = True
        else:
            if os.path.exists(self.checkpoint_path):
                if os.path.isfile(os.path.join(self.checkpoint_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    logger.info("Model restored from {}".format(latest_checkpoint))
                    is_restore = True
        if not is_restore:
            logger.info("Model started from random weights")

    def regularization_loss(self):
        reg_losses = []
        list_vars = [
            "cnn/conv1d_1/conv2d/kernel:0",
            "cnn/conv1d_2_1/conv2d/kernel:0",
            "cnn/conv1d_2_2/conv2d/kernel:0",
            "cnn/conv1d_2_3/conv2d/kernel:0",
            # "rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0",
            # "softmax_linear/dense/kernel:0",
        ]
        for v in tf.trainable_variables():
            if any(v.name in s for s in list_vars):
                reg_losses.append(tf.nn.l2_loss(v))
        if len(reg_losses):
            reg_losses = tf.multiply(tf.add_n(reg_losses, name="l2_loss"), self.config["l2_weight_decay"])
        else:
            reg_losses = 0
        return reg_losses


# ModelMod5 --> Use augmented sequence


class ModelMod6(ModelMod4):
    '''Use augmented sequence for the whole network and signals (horizontal shift) only for the CNN parts.'''

    def __init__(
        self,
        config,
        output_dir="./output",
        use_rnn=False,
        testing=False,
        use_best=False,
    ):
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")
        self.use_rnn = use_rnn

        # Placeholder
        with tf.variable_scope("placeholders") as scope:
            self.signals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1), name='signals')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')
            self.is_augment = tf.placeholder(dtype=tf.bool, shape=(), name='is_augment')

            if self.use_rnn:
                self.loss_weights = tf.placeholder(dtype=tf.float32, shape=(None, ), name='loss_weights')
                self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=(None, ), name='seq_lengths')

        # Monitor global step update
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Monitor the number of epochs passed
        self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

        # Build a network that receives inputs from placeholders
        all_nets = self.build_cnn()

        ## Supervised part - continue to RNN ##
        if not testing:
            # If augmented, slice the first part to feed to RNN
            net, ori_net, aug_net = tf.split(all_nets, num_or_size_splits=3, axis=0)
            net = tf.cond(
                self.is_augment,
                true_fn=lambda: net,
                false_fn=lambda: all_nets,
                name='cond_sup_part',
            )
        else:
            net = all_nets

        if self.use_rnn:
            # Check whether the corresponding config is given
            if "n_rnn_layers" not in self.config:
                raise Exception("Invalid config.")
            # Append the RNN if needed
            net = self.append_rnn(net)

        # Softmax linear
        net = nn.fc("softmax_linear", net, self.config["n_classes"], bias=0.0)

        # Outputs
        self.logits = net
        self.preds = tf.argmax(self.logits, axis=1)

        # Cross-entropy loss
        self.loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.logits,
            name="loss_ce_per_sample"
        )

        with tf.name_scope("loss_ce_mean") as scope:
            if self.use_rnn:
                # Weight by sequence
                loss_w_seq = tf.multiply(self.loss_weights, self.loss_per_sample)

                # Weight by class
                sample_weights = tf.reduce_sum(
                    tf.multiply(
                        tf.one_hot(indices=self.labels, depth=self.config["n_classes"]), 
                        np.asarray(self.config["class_weights"], dtype=np.float32)
                    ), 1
                )
                loss_w_class = tf.multiply(loss_w_seq, sample_weights)

                # Computer average loss scaled with the sequence length
                self.loss_ce = tf.reduce_sum(loss_w_class) / tf.reduce_sum(self.loss_weights)
            else:
                self.loss_ce = tf.reduce_mean(self.loss_per_sample)

        ## Augmentation part ##
        if not testing:
            # If augmented, uses the remaining parts to for augmented loss
            remain_nets = tf.concat([ori_net, aug_net], axis=0)
            # Softmax linear - compress to have `n_classes` values
            remain_nets = nn.fc("softmax_linear_aug", remain_nets, self.config["n_classes"], bias=0.0)
            self.ori_logits, self.aug_logits = tf.split(remain_nets, num_or_size_splits=2, axis=0)

            def _kl_divergence_with_logits(p_logits, q_logits):
                '''Ref: https://github.com/google-research/uda'''
                p = tf.nn.softmax(p_logits)
                log_p = tf.nn.log_softmax(p_logits)
                log_q = tf.nn.log_softmax(q_logits)

                kl = tf.reduce_sum(p * (log_p - log_q), -1)
                return kl

            unsup_aug_loss = _kl_divergence_with_logits(
                # Stop gradient -> treat as a fixed copy of the current params
                p_logits=tf.stop_gradient(self.ori_logits),
                q_logits=self.aug_logits
            )
            self.aug_loss = tf.cond(
                self.is_augment,
                true_fn=lambda: tf.multiply(tf.reduce_mean(unsup_aug_loss), self.config['aug_coeff']),
                false_fn=lambda: tf.constant(0.0), # dummy
                name='cond_aug_loss',
            )
        else:
            self.aug_loss = tf.constant(0.0)

        # Regularization loss
        self.reg_losses = self.regularization_loss()

        # Total loss
        self.loss = self.loss_ce + self.aug_loss + self.reg_losses

        # Metrics (used when we want to compute a metric from the output from minibatches)
        with tf.variable_scope("stream_metrics") as scope:
            self.metric_value_op, self.metric_update_op = contrib_metrics.aggregate_metric_map({
                "loss": tf.metrics.mean(values=self.loss),
                "accuracy": tf.metrics.accuracy(labels=self.labels, predictions=self.preds),
                "precision": tf.metrics.precision(labels=self.labels, predictions=self.preds),
                "recall": tf.metrics.recall(labels=self.labels, predictions=self.preds),
            })
            # Manually create reset operations of local vars
            metric_vars = contrib_slim.get_local_variables(scope=scope.name)
            self.metric_init_op = tf.variables_initializer(metric_vars)

        # Training outputs
        self.train_outputs = {
            "global_step": self.global_step,
            "train/loss": self.loss,
            "train/preds": self.preds,
            "train/stream_metrics": self.metric_update_op,
        }
        if self.use_rnn:
            self.train_outputs.update({
                "train/init_state": self.init_state,
                "train/final_state": self.final_state,
            })

        # Test outputs
        self.test_outputs = {
            "global_step": self.global_step,
            "test/loss": self.loss,
            "test/preds": self.preds,
        }
        if self.use_rnn:
            self.test_outputs.update({
                "test/init_state": self.init_state,
                "test/final_state": self.final_state,
            })

        # Tensoflow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        if not testing:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"))
            self.train_writer.add_graph(self.sess.graph)
            logger.info("Saved tensorboard graph to {}".format(self.train_writer.get_logdir()))

        # Optimizer
        if not testing:
            # self.lr = tf.train.exponential_decay(
            #     learning_rate=self.config["learning_rate_decay"],
            #     global_step=self.global_step,
            #     decay_steps=self.config["decay_steps"],
            #     decay_rate=self.config["decay_rate"],
            #     staircase=False,
            #     name="learning_rate"
            # )
            self.lr = tf.constant(self.config["learning_rate"], dtype=tf.float32)
            with tf.variable_scope("optimizer") as scope:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Pretraining
                    if not self.use_rnn:
                        self.train_step_op, self.grad_op = nn.adam_optimizer(
                            loss=self.loss,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            # learning_rate=self.config["learning_rate"],
                            learning_rate=self.lr,
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                        )
                    # Fine-tuning
                    else:
                        # Use different learning rates for CNN and RNN
                        self.train_step_op, self.grad_op = nn.adam_optimizer_clip(
                            loss=self.loss,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            # learning_rate=self.config["learning_rate"],
                            learning_rate=self.lr,
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                            clip_value=self.config["clip_grad_value"],
                        )

        # Initializer
        with tf.variable_scope("initializer") as scope:
            # tf.trainable_variables() or tf.global_variables()
            self.init_global_op = tf.variables_initializer(tf.global_variables())
            self.init_local_op = tf.variables_initializer(tf.local_variables())

        # Saver for storing variables
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Initialize variables
        self.run([self.init_global_op, self.init_local_op])

        # Restore variables (if possible)
        is_restore = False
        if use_best:
            if os.path.exists(self.best_ckpt_path):
                if os.path.isfile(os.path.join(self.best_ckpt_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.best_ckpt_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    logger.info("Best model restored from {}".format(latest_checkpoint))
                    is_restore = True
        else:
            if os.path.exists(self.checkpoint_path):
                if os.path.isfile(os.path.join(self.checkpoint_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    logger.info("Model restored from {}".format(latest_checkpoint))
                    is_restore = True
        if not is_restore:
            logger.info("Model started from random weights")

    def train_aug(self, minibatches, aug_minibatches):
        self.run(self.metric_init_op)
        start = timeit.default_timer()
        preds = []
        trues = []

        for batch, aug_batch in zip(minibatches, aug_minibatches):
            x, y, w, sl, re = batch
            aug_x, _, _, _, _ = aug_batch
            all_x = np.concatenate([x, x, aug_x], axis=0)

            feed_dict = {
                self.signals: all_x,
                self.labels: y,
                self.is_training: True,
                self.loss_weights: w,
                self.seq_lengths: sl,
                self.is_augment: True,
            }

            if re:
                # Initialize state of RNN
                state = self.run(self.init_state)

            # Carry the states from the previous batches through time
            for i, (c, h) in enumerate(self.init_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)

            # Buffer the final states
            state = outputs["train/final_state"]

            tmp_preds = np.reshape(outputs["train/preds"], (self.config["batch_size"], self.config["seq_length"]))
            tmp_trues = np.reshape(y, (self.config["batch_size"], self.config["seq_length"]))

            for i in range(self.config["batch_size"]):
                preds.extend(tmp_preds[i, :sl[i]])
                trues.extend(tmp_trues[i, :sl[i]])

        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs.update({
            "train/trues": trues,
            "train/preds": preds,
            "train/accuracy": acc,
            "train/f1_score": f1_score,
            "train/cm": cm,
            "train/duration": duration,
        })
        return outputs

    def evaluate_aug(self, minibatches):
        start = timeit.default_timer()
        losses = []
        preds = []
        trues = []

        for x, y, w, sl, re in minibatches:
            feed_dict = {
                self.signals: x,
                self.labels: y,
                self.is_training: False,
                self.loss_weights: w,
                self.seq_lengths: sl,
                self.is_augment: False,
            }

            if re:
                # Initialize state of RNN
                state = self.run(self.init_state)

            # Carry the states from the previous batches through time
            for i, (c, h) in enumerate(self.init_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            outputs = self.run(self.test_outputs, feed_dict=feed_dict)

            # Buffer the final states
            state = outputs["test/final_state"]

            losses.append(outputs["test/loss"])

            tmp_preds = np.reshape(outputs["test/preds"], (self.config["batch_size"], self.config["seq_length"]))
            tmp_trues = np.reshape(y, (self.config["batch_size"], self.config["seq_length"]))

            for i in range(self.config["batch_size"]):
                preds.extend(tmp_preds[i, :sl[i]])
                trues.extend(tmp_trues[i, :sl[i]])

        loss = np.mean(losses)
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs = {
            "test/trues": trues,
            "test/preds": preds,
            "test/loss": loss,
            "test/accuracy": acc,
            "test/f1_score": f1_score,
            "test/cm": cm,
            "test/duration": duration,
        }
        return outputs


# ModelMod7 --> Use augmented sequence and signals (horizontal shift) for the whole network.


class ModelMod8(ModelMod4):
    '''Remove dropout layer after LSTM, as there has already been a dropout wrapper.'''

    def append_rnn(self, inputs):
        # RNN
        with tf.variable_scope("rnn") as scope:
            # Reshape the input from (batch_size * seq_length, input_dim) to
            # (batch_size, seq_length, input_dim)
            input_dim = inputs.shape[-1].value
            seq_inputs = tf.reshape(inputs, shape=[-1, self.config["seq_length"], input_dim], name="reshape_seq_inputs")

            def _create_rnn_cell(n_units):
                """A function to create a new rnn cell."""
                cell = tf.contrib.rnn.LSTMCell(
                    num_units=n_units,
                    use_peepholes=True,
                    forget_bias=1.0,
                    state_is_tuple=True,
                )
                # Dropout wrapper
                keep_prob = tf.cond(self.is_training, lambda:tf.constant(0.5), lambda:tf.constant(1.0))
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
                return cell

            # LSTM
            cells = []
            for l in range(self.config["n_rnn_layers"]):
                cells.append(_create_rnn_cell(self.config["n_rnn_units"]))

            # Multiple layers of forward and backward cells
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)

            # Initial states
            self.init_state = multi_cell.zero_state(self.config["batch_size"], tf.float32)

            # Create rnn
            outputs, states = tf.nn.dynamic_rnn(
                cell=multi_cell,
                inputs=seq_inputs,
                initial_state=self.init_state,
                sequence_length=self.seq_lengths,
            )

            # Final states
            self.final_state = states

            # Concatenate the output from forward and backward cells
            net = tf.reshape(outputs, shape=[-1, self.config["n_rnn_units"]], name="reshape_nonseq_input")

            # net = tf.layers.dropout(net, rate=0.75, training=self.is_training, name="drop")

        return net

if __name__ == "__main__":
    from config import pretrain
    model = Model(config=pretrain, output_dir="./output/test", use_rnn=False)
    tf.reset_default_graph()
    from config import finetune
    model = Model(config=finetune, output_dir="./output/test", use_rnn=True)
