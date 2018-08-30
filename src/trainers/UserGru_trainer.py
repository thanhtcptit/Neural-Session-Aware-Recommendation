import sys
from time import time

import numpy as np
import tensorflow as tf

from src.base.base_train import BaseTrain
from src.data_loader.data_loader import DataLoader
from src.trainers.UserGru_evaluator import UserGruEval
from src.utils.qpath import CHECKPOINT_DIR

sys.path.append("../..")


class UserGruTrainer(BaseTrain):
    def __init__(self, sess, model, config, data_loader, logger=None):
        super(UserGruTrainer, self).__init__(
            sess, model, config, data_loader, logger)
        if config.dev_path is not None:
            self.dev_loader = DataLoader(config.dev_path, config)
            self.evaluator = UserGruEval(sess, model, config, self.dev_loader)
            self.best_acc = 0

        self.sess.run(tf.global_variables_initializer())

    def run_training(self):
        for epoch in range(self.config.num_epoch):
            start = time()
            self.data_loader.next_epoch()
            epoch_loss = self.train_epoch()

            print('++ Epoch: {} - Loss: {:.5f} - Time: {:.5f} ++'.format(
                  epoch, epoch_loss, time() - start))

            if self.config.dev_path is not None \
                    and epoch % self.config.eval_every == 0:
                acc, mrr = self.evaluator.run_evaluation()
                if acc[0] > self.best_acc:
                    self.best_acc = acc[0]
                    self.save(CHECKPOINT_DIR + self.config.name + '-best.ckpt')
                print('++ Evaluate result on dev set ++')
                for k, r, m in zip([5, 10, 20], acc, mrr):
                    print('Recall@{}: {:.4f}  -  MRR@{}: {:.4f}'.format(k, r, k, m))

    def train_epoch(self):
        losses = []
        while self.data_loader.has_next():
            start = time()
            loss, step = self.train_step()
            losses.append(loss)

            if step % self.config.display_every == 0:
                print('Step : {} - Loss: {:.5f} ' '- Time: {:.5f}'.format(
                    step, loss, time() - start))

            if step % self.config.save_every == 0:
                self.save(CHECKPOINT_DIR + self.config.name + '.ckpt')

        return np.mean(losses)

    def train_step(self):
        batch_data = self.data_loader.next_batch()
        feed_dict = {
            self.model.user: batch_data[:, :-1, 0],
            self.model.item: batch_data[:, :-1, 1],
            self.model.hour: batch_data[:, :-1, 2],
            self.model.day_of_week: batch_data[:, :-1, 3],
            self.model.month_period: batch_data[:, :-1, 4],
            self.model.next_items: batch_data[:, 1:, 1],
            self.model.keep_pr: self.config.keep_pr
        }
        _, batch_loss, step = self.sess.run(self.model.get_training_vars(),
                                            feed_dict=feed_dict)
        return batch_loss, step

    def save(self, path):
        save_path = self.saver.save(self.sess, path)
        print('++ Save model to {} ++'.format(save_path))

    def load(self, path):
        load_path = self.saver.restore(self.sess, path)
        print('++ Load model from {} ++'.format(load_path))
