import sys

import tensorflow as tf
from tensorflow.contrib.rnn import *

from src.base.base_model import BaseModel

sys.path.append("../..")


class UserGruModel(BaseModel):
    def __init__(self, config):
        super(UserGruModel, self).__init__(config)

        # Data stats
        self._num_users = config.num_users
        self._num_items = config.num_items
        self._max_length = config.max_length

        # Hyper parameters
        self._cell = config.cell
        self._entity_embedding = config.entity_embedding
        self._time_embedding = config.time_embedding
        self._hidden_units = config.hidden_units
        self._num_layers = config.num_layers

        # Input
        self._input_type = config.input

        self.print_info()

        # Placeholder
        self.user = tf.placeholder(tf.int32, shape=[None, self._max_length])
        self.item = tf.placeholder(tf.int32, shape=[None, self._max_length])
        self.hour = tf.placeholder(tf.int32, shape=[None, self._max_length])
        self.day_of_week = tf.placeholder(
            tf.int32, shape=[None, self._max_length])
        self.month_period = tf.placeholder(
            tf.int32, shape=[None, self._max_length])
        self.next_items = tf.placeholder(
            tf.int32, shape=[None, self.config.max_length])
        self.labels = tf.one_hot(depth=self.config.num_items + 1,
                                 indices=self.next_items, dtype=tf.int32)
        self.keep_pr = tf.placeholder(tf.float32)

        self.length = tf.reduce_sum(tf.sign(self.next_items), axis=1)
        self.global_step = tf.Variable(0, name="global_step",
                                       trainable=False)

        # Model variable
        self._E = {}
        self._embs = {}
        self._rnn_cell = None
        self._w_fc = None
        self._b_fc = None
        self._V = {}
        self._b = {}

        # Output
        self.loss = None
        self.optimizer = None
        self.train_op = None
        self._logits = None
        self._output_prob = None

        self.build_model()

    def print_info(self):
        print('--- Model info ---')
        print('- Num users: ', self._num_users)
        print('- Num items: ', self._num_items)
        print('- Input type: ', self._input_type)
        print('- Max session length: ', self._max_length)
        print('- Entity embedding: ', self._entity_embedding)
        print('- Time embedding: ', self._time_embedding)
        print('- Hidden unit: ', self._hidden_units)
        print('- Num layers: ', self._num_layers)
        print('- RNN cell: ', self._cell)

    def build_model(self):
        with tf.variable_scope('embeddings'):
            for x, y, k in zip([self._num_items + 1,
                                self._num_users + 1, 25, 8, 25],
                               [self._entity_embedding] * 2 +
                               [self._time_embedding] * 3,
                               ['i', 'u', 'h', 'd', 'm']):
                self._E[k] = tf.get_variable(shape=[x, y],
                                             name='E' + k, dtype=tf.float32)
        for v, k in zip([self.item, self.user, self.hour,
                         self.day_of_week, self.month_period],
                        ['i', 'u', 'h', 'd', 'm']):
            self._embs[k] = tf.nn.embedding_lookup(self._E[k], v)

        self._embs['u'] = tf.nn.dropout(self._embs['u'], self.keep_pr)
        self._embs['i'] = tf.nn.dropout(self._embs['i'], self.keep_pr)

        with tf.variable_scope('rnn-cell'):
            if self._cell == 'gru':
                self._rnn_cell = MultiRNNCell([GRUCell(self._hidden_units)
                                              for _ in range(self._num_layers)])
            elif self._cell == 'lstm':
                self._rnn_cell = MultiRNNCell([LSTMCell(self._hidden_units)
                                               for _ in range(self._num_layers)])
            else:
                self._rnn_cell = MultiRNNCell([RNNCell(self._hidden_units)
                                               for _ in range(self._num_layers)])

        output_states, _ = tf.nn.dynamic_rnn(self._rnn_cell, self._embs['i'],
                                             sequence_length=self.length,
                                             dtype=tf.float32)
        if self._input_type == 'concat':
            final_state = tf.reshape(
                tf.concat([output_states, self._embs['u']], -1),
                [-1, self._hidden_units + self._entity_embedding])
        elif self._input_type == 'concat-context':
            final_state = tf.reshape(
                tf.concat([output_states, self._embs['u'], self._embs['h'],
                           self._embs['d'], self._embs['m']], -1),
                [-1, self._hidden_units
                 + self._entity_embedding
                 + 3 * self._time_embedding])
        elif self._input_type == 'attention':
            final_state = self._attention(output_states, self._embs['u'])
        else:
            final_state = self._attention_context(
                output_states, self._embs['u'], self._embs['h'],
                self._embs['d'], self._embs['m'])

        with tf.name_scope('softmax'):
            last_dim = self._hidden_units + self._entity_embedding
            if 'context' in self._input_type:
                last_dim += + 3 * self._time_embedding
            self._w_fc = tf.get_variable(shape=[last_dim, self._num_items + 1],
                                         name='w_fc', dtype=tf.float32)
            self._b_fc = tf.get_variable(shape=[self._num_items + 1],
                                         name='b_fc', dtype=tf.float32)

        self._logits = tf.matmul(final_state, self._w_fc) + self._b_fc
        self._output_prob = tf.nn.softmax(self._logits)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.labels, logits=self._logits)
        self.loss = tf.reduce_mean(self.loss)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def _attention_context(self, item, user, hour, day, month):
        with tf.name_scope('attention'):
            self._V = {}
            self._b = {}
            for x, k in zip([self._hidden_units, self._entity_embedding]
                            + [self._time_embedding] * 3,
                            ['i', 'u', 'h', 'd', 'm']):
                self._V[k] = tf.get_variable(shape=[x],
                                             name='V' + k, dtype=tf.float32)

            for k in ['i', 'u', 'h', 'd', 'm']:
                self._b[k] = tf.get_variable(shape=[], name='b' + k,
                                             dtype=tf.float32)

        alpha = []
        for x, k in zip([item, user, hour, day, month],
                        ['i', 'u', 'h', 'd', 'm']):
            alpha.append(tf.sigmoid(tf.reduce_sum(
                tf.cast(x, tf.float32) * self._V[k], axis=2) + self._b[k]))

        attention_w = []
        for t in range(self._max_length):
            wt = []
            for i in range(5):
                wt.append(alpha[i][:, t])
            sum_exp = tf.reduce_sum(tf.exp(wt), axis=0)
            attention_w.append([tf.exp(w_) / sum_exp for w_ in wt])

        attention_w = tf.transpose(tf.stack(attention_w), [2, 0, 1])
        final_input = []
        for i, x in enumerate([item, user, hour, day, month]):
            final_input.append(tf.expand_dims(attention_w[:, :, i], dim=2) * x)
        return tf.reshape(tf.concat(final_input, -1),
                          [-1, self._hidden_units + self._entity_embedding
                           + 3 * self._time_embedding])

    def _attention(self, item, user):
        with tf.name_scope('attention'):
            self._V = {}
            self._b = {}
            for x, k in zip([self._entity_embedding] * 2, ['i', 'u']):
                self._V[k] = tf.get_variable(shape=[x],
                                             name='V' + k, dtype=tf.float32)
            for k in ['i', 'u']:
                self._b[k] = tf.get_variable(shape=[], name='b' + k, dtype=tf.float32)

        alpha = []
        for x, k in zip([item, user], ['i', 'u']):
            alpha.append(tf.sigmoid(tf.reduce_sum(
                tf.cast(x, tf.float32) * self._V[k], axis=2) + self._b[k]))

        attention_w = []
        for t in range(self._max_length):
            wt = []
            for i in range(2):
                wt.append(alpha[i][:, t])
            sum_exp = tf.reduce_sum(tf.exp(wt), axis=0)
            attention_w.append([tf.exp(w_) / sum_exp for w_ in wt])

        attention_w = tf.transpose(tf.stack(attention_w), [2, 0, 1])
        final_input = []
        for i, x in enumerate([item, user]):
            final_input.append(tf.expand_dims(attention_w[:, :, i], dim=2) * x)
        return tf.reshape(tf.concat(final_input, -1),
                          [-1, self._hidden_units + self._entity_embedding])

    def get_training_vars(self):
        return self.train_op, self.loss, self.global_step

    def get_output(self):
        return self._output_prob
