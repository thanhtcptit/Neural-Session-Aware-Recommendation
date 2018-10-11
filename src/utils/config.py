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
        self.fusion_type = 'post'

        # Data path
        self.train_path = PROCESSED_DATA_DIR + 'clean-lastfm-train'
        self.test_path = PROCESSED_DATA_DIR + 'clean-lastfm-test'
        self.data_stats = PROCESSED_DATA_DIR + 'clean-lastfm-train-metadata'

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
        self.fusion_type = args.fusion_type

        # Data path
        self.train_path = PROCESSED_DATA_DIR + args.train_file
        if args.test_file is not None:
            self.test_path = PROCESSED_DATA_DIR + args.test_file
        else:
            self.test_path = None
        self.data_stats = PROCESSED_DATA_DIR + args.train_file + '-metadata'

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

        self.get_data_stats()

    def save_model_config(self):
        with open(CHECKPOINT_DIR + self.name + '_config.txt', 'w') as f:
            f.write(str(self.input) + '\n')
            f.write(str(self.fusion_type) + '\n')
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
        with open(CHECKPOINT_DIR + name + '_config.txt', 'r') as f:
            data = f.read().split('\n')
            self.input, self.fusion_type, self.cell,\
                self.num_layers, self.entity_embedding,\
                self.time_embedding, self.hidden_units = data
        self.num_layers = int(self.num_layers)
        self.entity_embedding = int(self.entity_embedding)
        self.time_embedding = int(self.time_embedding)
        self.hidden_units = int(self.hidden_units)

    def get_data_stats(self):
        with open(self.data_stats, 'r') as f:
            data = f.read().split('\n')
        self.num_items, self.num_users, self.max_length = list(map(int, data))
