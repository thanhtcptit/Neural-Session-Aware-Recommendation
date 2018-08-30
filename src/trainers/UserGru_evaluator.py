import numpy as np

from src.base.base_eval import BaseEval
from src.utils.qpath import CHECKPOINT_DIR


class UserGruEval(BaseEval):
    def __init__(self, sess, model, config, data_loader, logger=None, init_graph=False):
        super(UserGruEval, self).__init__(
            sess, model, config, data_loader, logger, init_graph)
        self.acc = np.array([0.] * 3, dtype=np.float32)
        self.mrr = np.array([0.] * 3, dtype=np.float32)
        self.num_events_eval = 0

    def load(self, path):
        self.saver.restore(self.sess, path)

    def calculate_ranks(self, _pr, y_true):
        y_true = np.reshape(y_true, [-1])
        rows_idx = [i for i in range(len(y_true)) if y_true[i] != 0]
        mask_rows_idx = [[i] for i in range(len(y_true)) if y_true[i] != 0]
        mask_cols_idx = [[j] for j in y_true if j != 0]
        ranks = (_pr[rows_idx, :] >
                 _pr[mask_rows_idx, mask_cols_idx]).sum(axis=1) + 1
        return ranks, len(rows_idx)

    def evaluate(self, ranks, top):
        count_true = [0.] * len(top)
        rr = [0.] * len(top)

        for i, n in enumerate(top):
            true_predict = ranks <= n
            count_true[i] += true_predict.sum()
            rr[i] += (1. / ranks[true_predict]).sum()
        return count_true, rr

    def run_evaluation(self):
        self.data_loader.next_epoch()
        while self.data_loader.has_next():
            self.eval_step()

        self.acc /= self.num_events_eval
        self.mrr /= self.num_events_eval

        return self.acc, self.mrr

    def eval_step(self):
        batch_data = self.data_loader.next_batch()

        feed_dict = {
            self.model.user: batch_data[:, :-1, 0],
            self.model.item: batch_data[:, :-1, 1],
            self.model.hour: batch_data[:, :-1, 2],
            self.model.day_of_week: batch_data[:, :-1, 3],
            self.model.month_period: batch_data[:, :-1, 4],
            self.model.keep_pr: 1
        }
        logits, pr = self.sess.run(self.model.get_output(), feed_dict=feed_dict)
        assert len(pr) == 1
        batch_ranks, num_events = \
            self.calculate_ranks(pr[0], batch_data[:, 1:, 1])
        batch_cp, batch_rr = self.evaluate(batch_ranks, [5, 10, 20])
        self.acc += batch_cp
        self.mrr += batch_rr
        self.num_events_eval += num_events
