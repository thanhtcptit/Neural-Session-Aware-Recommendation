import sys
sys.path.append('../..')

import numpy as np

from time import time

from src.base.base_eval import BaseEval
from src.utils.qpath import *


class UserGruEval(BaseEval):
    def __init__(self, sess, model, config,
                 data_loader, logger=None, init_graph=False):
        super(UserGruEval, self).__init__(
            sess, model, config, data_loader, logger, init_graph)

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
            self.model.day_of_week: session[:, :-1, 3],
            self.model.month_period: session[:, :-1, 4],
            self.model.next_items: session[:, 1:, 1],
            self.model.keep_pr: 1
        }
        pr = self.sess.run(self.model.get_output(), feed_dict=feed_dict)
        assert len(pr) != 1
        pr = pr[pos]
        top_id = np.argpartition(pr, -10)[-10:]
        top_id = top_id[np.argsort(pr[top_id])]
        with open('/tmp/res.txt', 'w') as f:
            f.write(' '.join(map(str, top_id.tolist())))

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
                    self.model.day_of_week: tmp[:, :-1, 3],
                    self.model.month_period: tmp[:, :-1, 4],
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

    def run_evaluation(self):
        self.data_loader.next_epoch()
        acc = np.array([0.] * 2, dtype=np.float32)
        mrr = np.array([0.] * 2, dtype=np.float32)
        num_events_eval = 0
        while self.data_loader.has_next():
            batch_cp, batch_rr, batch_events = self.eval_step()
            acc += batch_cp
            mrr += batch_rr
            num_events_eval += batch_events

        acc /= num_events_eval
        mrr /= num_events_eval

        return acc, mrr

    def eval_step(self):
        batch_data = self.data_loader.next_batch()

        feed_dict = {
            self.model.user: batch_data[:, :-1, 0],
            self.model.item: batch_data[:, :-1, 1],
            self.model.day_of_week: batch_data[:, :-1, 3],
            self.model.month_period: batch_data[:, :-1, 4],
            self.model.next_items: batch_data[:, 1:, 1],
            self.model.keep_pr: 1
        }
        pr = self.sess.run(self.model.get_output(), feed_dict=feed_dict)
        assert len(pr) != 1
        batch_ranks, num_events = \
            self.calculate_ranks(pr, batch_data[:, 1:, 1])
        batch_cp, batch_rr = self.evaluate(batch_ranks, [5, 20])

        return batch_cp, batch_rr, num_events
