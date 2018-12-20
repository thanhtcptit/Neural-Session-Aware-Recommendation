import sys
sys.path.append('/home/nero/py/rnn-for-resys/src/')
sys.path.append('/home/nero/py/rnn-for-resys/')

import tensorflow as tf
import numpy as np

from src.models.UserGru import UserGruModel
from src.utils.qpath import *
from src.utils.config import Args
from src.trainers.UserGru_evaluator import UserGruEval


def get_tensorflow_session():
    config = tf.ConfigProto(device_count={'GPU': 1})
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def run_predict(args):
    session = [[]]
    with open('/tmp/sess.txt') as f:
        for line in f:
            u, i = line.split(' ')
            session[0].append([int(u), int(i), 0, 0, 0])
    pos = len(session[0]) - 1
    session[0].append([1, 1, 0, 0, 0])
    if len(session[0]) >= 11:
        session[0] = session[0][-11:]
        pos = 9
    else:
        tmp = len(session[0])
        for i in range(11 - tmp):
            session[0].append([0, 0, 0, 0, 0])

    session = np.array(session)
    args.load_model_config()
    sess = get_tensorflow_session()
    model = UserGruModel(args)

    evaluator = UserGruEval(sess, model, args, None)
    evaluator.load(CHECKPOINT_DIR + args.name + '.ckpt')
    evaluator.run_predict(session, pos)
    # evaluator.run_test()
    print('done')


if __name__ == '__main__':
    try:
        args = Args()
        args.name = 'UserGru1-best'
        args.mode = 'predict'
        args.data_stats = PROCESSED_DATA_DIR + 'clean-train-metadata'
        args.get_data_stats()
    except Exception as e:
        print("missing or invalid arguments %s" % e)
        exit(0)
    run_predict(args)
