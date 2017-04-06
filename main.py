#####################################################
#
#
#   project - tensorflow boilerplate
#
#
#####################################################

import numpy as np
import tensorflow as tf

from utils.utils import *
from model import Config
from model import Nueral_Net_Model as nnm


class ModelName(nnm):
    """ Model Description.
    """

#    def __init__(self, config):
#        self.c = config

    def add_placeholders(self):
        # Add placeholders for data input in models.

#        self.input_message_placeholder = tf.placeholder(dtype=tf.float32, \
#					shape=[self.c.batch_size, self.c.plain_text_length])
        pass

    def add_training_vars(self):
        # Add the weights for the model.
        # An example is shown below, you can use the layer_and_weights method in utils.
#        # peer 3 weights.
#        with tf.name_scope('peer_adv_fc_layer'):
#            self.peer_adv_fc_weight = self.lw.get_weights('peer_adv_fc_weight', [self.c.plain_text_length, self.c.plain_text_length], 'xavier')
#            self.peer_adv_fc_bias = self.lw.get_weights('peer_adv_fc_bias', [self.c.plain_text_length], \
#                                        initializer='constant', constant=0.1)
#            self.add_to_summaries([self.peer_adv])
        pass


    def add_training_op(self):
        # Add training Operation to the model.
        # An example is shown below.

#        with tf.name_scope('input_training'):
#            with tf.name_scope('peer_1'):
#                with tf.name_scope('peer_1_fc_layer'):
#                    self.peer_1_fc_out = self.lw.FC_layer(self.input_message_placeholder,\
#                                                self.peer_1_fc_weight, self.peer_1_fc_bias)
#                    self.peer_1_fc_out = tf.nn.sigmoid(self.peer_1_fc_out)
#                    self.add_to_summaries([self.peer_1_fc_out])
#                    self.peer_1_fc_out = tf.expand_dims(self.peer_1_fc_out, axis=2)
#                with tf.name_scope('peer_1_conv1_layer'):
#                    self.peer_1_conv1_out = self.lw.conv1d_layer(self.peer_1_fc_out, self.peer_1_conv1_weight, \
#                                                            self.peer_1_conv1_bias, strides=1)
#                    self.peer_1_conv1_out = tf.nn.sigmoid(self.peer_1_conv1_out)
#                    self.add_to_summaries([self.peer_1_conv1_out])
#                with tf.name_scope('peer_1_conv2_layer'):
#                    self.peer_1_conv2_out = self.lw.conv1d_layer(self.peer_1_conv1_out, self.peer_1_conv2_weight,\
#                                                self.peer_1_conv2_bias, strides=2)
#                    self.peer_1_conv2_out = tf.nn.sigmoid(self.peer_1_conv2_out)
#                    self.add_to_summaries([self.peer_1_conv2_out])
#                with tf.name_scope('peer_1_conv3_layer'):
#                    self.peer_1_conv3_out = self.lw.conv1d_layer(self.peer_1_conv2_out, self.peer_1_conv3_weight,\
#                                                self.peer_1_conv3_bias, strides=1)
#                    self.peer_1_conv3_out = tf.nn.sigmoid(self.peer_1_conv3_out)
#                    self.add_to_summaries([self.peer_1_conv3_out])
#                with tf.name_scope('peer_1_conv4_layer'):
#                    self.peer_1_conv4_out = self.lw.conv1d_layer(self.peer_1_conv3_out, self.peer_1_conv4_weight,\
#                                                self.peer_1_conv4_bias, strides=1)
#                    self.peer_1_conv4_out = tf.nn.tanh(self.peer_1_conv4_out)
#                    self.peer_1_conv4_out = tf.squeeze(self.peer_1_conv4_out)
#                    self.add_to_summaries([self.peer_1_conv4_out])
#                ## fc layer for key part 1.
#                with tf.name_scope('peer_1_key_op'):
#                    self.peer_1_fc_key_out = self.lw.FC_layer(self.peer_1_conv4_out, self.peer_1_private_weight,\
#                                                        self.peer_1_private_bias)
#                    self.peer_1_fc_key_out = tf.nn.sigmoid(self.peer_1_fc_key_out)
#                    self.add_to_summaries([self.peer_1_fc_key_out])
#            #### ---------- peer 1 output = self.peer_1_out.
        pass


    def add_loss_op(self):
        abss_actual = tf.abs(tf.subtract(self.peer_2_out, self.input_message_placeholder))
        with tf.name_scope('loss_train_keys'):
            self.loss_train = tf.reduce_mean(abss_actual)
            self.variable_summaries(self.loss_train)
        with tf.name_scope('actual_bits_wrong'):
            self.bits_wrong_train = tf.reduce_mean(tf.reduce_sum(abss_actual, axis=1))
            self.variable_summaries(self.bits_wrong_train)
#        abss_adv = tf.abs(tf.subtract(self.peer_adv_conv4_out, self.input_message_placeholder))
#        with tf.name_scope('loss_adversarial_network'):
#            self.bits_wrong_adv = tf.reduce_mean(tf.reduce_sum(abss_adv, axis=1))
#        with tf.name_scope('actual_loss'):
#            self.loss_actual = self.loss_train - self.bits_wrong_adv
        pass


    def add_optimizers(self):
        # Filter the training variables.
        # A really nice example with multiple optimizers.

#        self.training_vars = tf.trainable_variables()
#        self.trainers_optimizer_1 = [var for var in self.training_vars \
#                                    if 'key' and 'peer_2' in var.name]
#        self.trainers_optimizer_2 = [var for var in self.training_vars \
#                                    if not 'peer_adv' in self.training_vars]
#
#        self.optimizer_1 = tf.train.AdamOptimizer(self.c.lr).minimize(self.loss_actual, \
#                                    var_list=self.trainers_optimizer_1)
#        self.optimizer_2 = tf.train.AdamOptimizer(self.c.lr).minimize(self.bits_wrong_adv, \
#                                    var_list=self.trainers_optimizer_2)

 
    def run_batch(self, session, batch_input, is_summary=False):
        # This method is optimized for the summary writer, you can periodically
        # pass true for is_summary arg. This would write the summaries to disk.
        # Also remember to declare the feed_dict properly.

#        feed_dict = {
#            self.input_message_placeholder: batch_input_message
#        }

#        if not is_summary:
#            loss_total, bits_wrong_1, _op1 = \
#                                session.run([self.loss_train, self.bits_wrong_train, \
#                                self.optimizer_1], feed_dict=feed_dict)
#        else:
#            merge = tf.summary.merge_all()
#            return session.run(merge, feed_dict=feed_dict)


def main(debug=True):

    config = Config()
    with tf.Graph().as_default():
        lw = layers_and_weights()
        customs = {
            'lw': lw
        }

        if debug:
            config.num_batches = 51
            config.num_epochs = 1

        _model = ModelName(config, **customs)
        init = tf.global_variables_initializer()
        writer = tf.summary.FileWriter('./tf_summary/1')

        with tf.Session() as sess:
            sess.run(init)
            writer.add_graph(sess.graph)

            for epoch in range(config.num_epochs):
                for batches in range(config.num_batches):
                    batch_message = DataClass.get_batch_sized_data(config.batch_size, \
                        config.plain_text_length)
                    if batches % 50 == 0:
                        # write summaries.
                        s = p2p_model.run_batch(sess, batch_message, True)
                        print("Summarizing - " + str((config.num_batches * epoch) + batches))
                        writer.add_summary(s, (config.num_batches * epoch) + batches)
                    else:
                        res = p2p_model.run_batch(sess, batch_message, batch_key)
    if not debug:
        print(res)
    return 0


if __name__ == '__main__':
    debug = False
    main(debug)