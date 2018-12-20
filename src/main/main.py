import sys
sys.path.append('../..')

import argparse
import tensorflow as tf
from tensorflow.python.client import device_lib

from src.data_loader.data_loader import DataLoader
from src.models.UserGru import UserGruModel
from src.trainers.UserGru_evaluator import UserGruEval
from src.trainers.UserGru_trainer import UserGruTrainer
from src.utils.config import Args
from src.utils.qpath import *


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    print([x.name for x in local_device_protos if x.device_type == 'GPU'])
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_tensorflow_session():
    config = tf.ConfigProto(device_count={'GPU': 1})
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def _parse_cmd():
    parser = argparse.ArgumentParser()
    # Running mode
    parser.add_argument('--mode', choices=['train', 'test'],
                        default='train')
    parser.add_argument("--name", type=str, default='UserGru')
    parser.add_argument('--input', choices=['concat', 'concat-context',
                                            'mul', 'mul-ff', 'cf',
                                            'attention', 'attention-sum',
                                            'attention-context',
                                            'attention-ew',
                                            'attention-global'],
                        default='concat')
    parser.add_argument('--fusion_type', choices=['pre', 'post'],
                        default='post')

    # Path
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str,
                        default=None)

    # Hyper params
    parser.add_argument('--cell', choices=['lstm', 'gru', 'rnn'],
                        default='gru')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--entity_emb', type=int, default=100)
    parser.add_argument('--time_emb', type=int, default=5)
    parser.add_argument('--hidden_units', type=int, default=100)

    # Learning params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--keep_pr', type=float, default=0.5)
    parser.add_argument('--num_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=50)

    # Logging & Summary
    parser.add_argument('--display_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=1)
    return parser.parse_args()


def run_training(args):
    sess = get_tensorflow_session()
    model = UserGruModel(args)

    train_loader = DataLoader(args.train_path, args)
    trainer = UserGruTrainer(sess, model, args, train_loader)

    if os.path.exists(CHECKPOINT_DIR + args.name + '.ckpt.index'):
        while True:
            print('Already exist a model with the same name.'
                  'Restore the old model?')
            choose = input('(y/n): ')
            if choose != 'y' and choose != 'n':
                print('Wrong option')
            else:
                break
        if choose == 'y':
            trainer.load(os.path.join(CHECKPOINT_DIR, args.name + '.ckpt'))
            args.load_model_config()

    args.save_model_config()
    trainer.run_training()


def run_evaluation(args):
    args.load_model_config()
    sess = get_tensorflow_session()
    model = UserGruModel(args)

    test_loader = DataLoader(args.test_path, args)
    evaluator = UserGruEval(sess, model, args, test_loader)
    evaluator.load(CHECKPOINT_DIR + args.name + '.ckpt')
    acc, mrr = evaluator.run_evaluation()
    print('++ Evaluate result on test set ++')
    for k, r, m in zip([5, 20], acc, mrr):
        print('Recall@{}: {}  -  MRR@{}: {}'.format(k, r, k, m))


if __name__ == '__main__':
    try:
        args = Args()
        args.parse_args(_parse_cmd())
    except Exception as e:
        print("missing or invalid arguments %s" % e)
        exit(0)
    if args.mode == 'train':
        run_training(args)
    else:
        run_evaluation(args)
