import sys
sys.path.append('../..')

import numpy as np
import tensorflow as tf

from time import time

from src.utils.qpath import *


class UserGruPredict():
    def __init__(self, sess, model, config):
        self.config = config
        self.model = model
        self.sess = sess
        self.saver = tf.train.Saver()

    def load(self, path):
        self.saver.restore(self.sess, path)
        print('++ Load model from {} ++'.format(path))

    @staticmethod
    def calculate_ranks(_pr, y_true):
        y_true = np.reshape(y_true, [-1])
        rows_idx = [i for i in range(len(y_true)) if y_true[i] != 0]
        mask_rows_idx = [[i] for i in range(len(y_true)) if y_true[i] != 0]
        mask_cols_idx = [[j] for j in y_true if j != 0]
        ranks = (_pr[rows_idx, :] >
                 _pr[mask_rows_idx, mask_cols_idx]).sum(axis=1) + 1
        return ranks, len(rows_idx)

    @staticmethod
    def evaluate(ranks, top):
        count_true = [0.] * len(top)
        rr = [0.] * len(top)

        for i, n in enumerate(top):
            true_predict = ranks <= n
            count_true[i] += true_predict.sum()
            rr[i] += (1. / ranks[true_predict]).sum()
        return count_true, rr

    def run_predict(self, session, pos):
        feed_dict = {
            self.model.user: session[:, :-1, 0],
            self.model.item: session[:, :-1, 1],
            self.model.day_of_week: session[:, :-1, 2],
            self.model.month_period: session[:, :-1, 3],
            self.model.next_items: session[:, 1:, 1],
            self.model.keep_pr: 1
        }
        pr, attention = self.sess.run([self.model.get_output(),
                                       self.model.get_attention_weight()],
                                      feed_dict=feed_dict)
        assert len(pr) != 1
        pr = pr[pos]
        current_item = session[0][pos][1]
        # print(session)
        # print(attention)

        print('===================')
        if 'context' in self.config.input:
            print('Item: ', attention[0][pos][0])
            print('User: ', attention[0][pos][1])
            print('Day of week: ', attention[0][pos][2])
            print('Half month: ', attention[0][pos][3])
        else:
            print('Item attention: ', attention[0][0][pos][0])
            print('User attention: ', attention[1][0][pos][0])

        top_id = np.argpartition(pr, -12)[-12:]
        top_id = top_id[np.argsort(pr[top_id])[::-1]]
        top_id = list(top_id)
        if 0 in top_id:
            del top_id[top_id.index(0)]
        if current_item in top_id:
            del top_id[top_id.index(current_item)]

        return top_id[:10]

    def run_test(self):
        pos = 0
        session = [[]]
        with open(PROCESSED_DATA_DIR + 'clean-dev') as f:
            for line in tqdm(f):
                if '-' in line:
                    session = [[]]
                    pos = 0
                    continue

                u, i, *_ = line.strip().split(',')
                session[0].append([u, i, 0, 0, 0])
                if len(session[0]) == 1:
                    continue

                tmp = [session[0][:]]
                l = len(tmp[0])
                for i in range(11 - l):
                    tmp[0].append([0, 0, 0, 0, 0])
                tmp = np.array(tmp)

                feed_dict = {
                    self.model.user: tmp[:, :-1, 0],
                    self.model.item: tmp[:, :-1, 1],
                    self.model.day_of_week: tmp[:, :-1, 2],
                    self.model.month_period: tmp[:, :-1, 3],
                    self.model.next_items: tmp[:, 1:, 1],
                    self.model.keep_pr: 1
                }
                pr = self.sess.run(
                    self.model.get_output(), feed_dict=feed_dict)
                assert len(pr) != 1
                pr = pr[pos]
                pos += 1
                top_id = np.argpartition(pr, -10)[-10:]
                top_id = top_id.tolist()
                if session[0][pos][1] in top_id:
                    print(session[0])

    def eval_step(self):
        batch_data = self.data_loader.next_batch()

        feed_dict = {
            self.model.user: batch_data[:, :-1, 0],
            self.model.item: batch_data[:, :-1, 1],
            self.model.day_of_week: batch_data[:, :-1, 2],
            self.model.month_period: batch_data[:, :-1, 3],
            self.model.next_items: batch_data[:, 1:, 1],
            self.model.keep_pr: 1
        }
        pr = self.sess.run(self.model.get_output(), feed_dict=feed_dict)
        assert len(pr) != 1
        batch_ranks, num_events = \
            self.calculate_ranks(pr, batch_data[:, 1:, 1])
        batch_cp, batch_rr = self.evaluate(batch_ranks, [5, 20])

        return batch_cp, batch_rr, num_events
