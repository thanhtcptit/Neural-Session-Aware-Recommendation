import sys
sys.path.append('../..')  # noqa

import time
import os
import grpc
import argparse
import tensorflow as tf
import numpy as np

from concurrent import futures

import resys_pb2
import resys_pb2_grpc

from src.utils.qpath import CHECKPOINT_DIR
from src.utils.config import Args
from src.main.main import _parse_cmd, get_tensorflow_session
from src.trainers.UserGru_predict import UserGruPredict
from src.models.UserGru import UserGruModel
from src.data.preprocess import extract_time_context_raw


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class ResysServicer(resys_pb2_grpc.ResysServicer):
    def __init__(self, config):
        model = UserGruModel(config)
        sess = get_tensorflow_session()
        self.resys = UserGruPredict(sess, model, config)
        self.resys.load(CHECKPOINT_DIR + config.name + '.ckpt')

    def get_items_iterator(self, items):
        for i in items:
            yield resys_pb2.Item(id=i)

    def GenerateRecommend(self, request_iterator, context):
        events = []
        try:
            for event in request_iterator:
                day, half_month = extract_time_context_raw(event.date)
                events.append([event.user, event.item, day, half_month])

            pos = len(events) - 1
            events.append([1, 1, 0, 0])
            if len(events) >= 11:
                events = events[-11:]
                pos = 9
            else:
                tmp = len(events)
                for i in range(11 - tmp):
                    events.append([0, 0, 0, 0])

            # print(events)
            events = [events]
            events = np.array(events)
            rec_items = self.resys.run_predict(events, pos)
            return self.get_items_iterator(rec_items)
        except Exception as e:
            print(e)


def start(server, config):
    resys_pb2_grpc.add_ResysServicer_to_server(
        ResysServicer(config), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('Service start')


def serve(config):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    start(server, config)
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


def main():
    args = Args()
    args.parse_args(_parse_cmd())
    # args.name = 'UserAGru-context-best'
    args.load_model_config()

    serve(args)


if __name__ == '__main__':
    main()
