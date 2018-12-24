from __future__ import print_function

import random

import grpc
import numpy as np
import time
import os

import resys_pb2
import resys_pb2_grpc


def generate_session(events):
    for event in events:
        yield resys_pb2.Event(
            user=event[0], item=event[1], date=event[2])


def call_GenerateRecommend(stub, events):
    response = stub.GenerateRecommend(get_Session(events))
    if response.status == 0:
        print('Error')
    else:
        pass
    return response


def run():
    events = [[1, 1, '2018-08-11 10:15:30'], [2, 1, '2018-11-16 10:15:30']]
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = resys_pb2_grpc.ResysStub(channel)
        try:
            it = stub.GenerateRecommend(generate_session(events))
            for r in it:
                print(r.id)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    run()
