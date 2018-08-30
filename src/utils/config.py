import json
import os
import sys
from easydict import EasyDict

from src.utils.qpath import *

sys.path.append("../..")


class Args:
    def __init__(self):
        # Running mode
        self.mode = 'train'
        self.name = 'GRU'
        self.input = 'concat'

        # Data path
        self.train_path = PROCESSED_DATA_DIR + 'clean-lastfm-train'
        self.dev_path = PROCESSED_DATA_DIR + 'clean-lastfm-dev'
        self.test_path = PROCESSED_DATA_DIR + 'clean-lastfm-test'
        self.config_path = PROCESSED_DATA_DIR + 'clean-lastfm-train-metadata'

        # Data stats
        self.num_users = None
        self.num_items = None
        self.max_length = None

        # Hyper params
        self.cell = 'GRU'
        self.num_layers = 1
        self.entity_embedding = 100
        self.time_embedding = 5
        self.hidden_units = 100

        # Learning params
        self.learning_rate = 0.001
        self.keep_pr = 1
        self.num_epoch = 30
        self.batch_size = 50

        # Logging
        self.display_every = 500
        self.save_every = 10000
        self.eval_every = 1

    def parse_args(self, args):
        # Running mode
        self.mode = args.mode
        self.name = args.name
        self.input = args.input

        # Data path
        self.train_path = PROCESSED_DATA_DIR + args.train_file
        self.dev_path = PROCESSED_DATA_DIR + args.dev_file
        self.test_path = PROCESSED_DATA_DIR + args.test_file
        self.config_path = PROCESSED_DATA_DIR + args.train_file + '-metadata'

        # Hyper params
        self.cell = args.cell
        self.num_layers = args.num_layers
        self.entity_embedding = args.entity_emb
        self.time_embedding = args.time_emb
        self.hidden_units = args.hidden_units

        # Learning params
        self.learning_rate = args.lr
        self.keep_pr = args.keep_pr
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        # Logging
        self.display_every = args.display_every
        self.save_every = args.save_every
        self.eval_every = args.eval_every

        self.get_data_config()

    def save_model_config(self):
        with open(CHECKPOINT_DIR + self.name + '_config.txt', 'w') as f:
            f.write(str(self.input) + '\n')
            f.write(str(self.cell) + '\n')
            f.write(str(self.num_layers) + '\n')
            f.write(str(self.entity_embedding) + '\n')
            f.write(str(self.time_embedding) + '\n')
            f.write(str(self.hidden_units))

    def load_model_config(self):
        if self.name.rfind('-best') == -1:
            name = self.name
        else:
            name = self.name[:self.name.rfind('-')]
        with open(CHECKPOINT_DIR + name + '_config', 'r') as f:
            data = map(int, f.read().split('\n'))
            self.input, self.cell, self.num_layers, self.entity_embedding, \
                self.time_embedding, self.hidden_units = data

    def get_data_config(self):
        with open(self.config_path, 'r') as f:
            data = f.read().split('\n')[:-1]
        self.num_items, self.num_users, self.max_length = list(map(int, data))
