import numpy as np


class DataLoader(object):
    def __init__(self, path, config):
        self._path = path
        self._max_length = config.max_length
        self._batch_size = config.batch_size
        self._batch_index = -1
        self._data = None
        self._num_events = None
        self._num_events_eval = None
        self._num_batch = None

        self.load_data()

    def load_data(self):
        self._data = []
        self._num_events = 0
        session = []
        with open(self._path, 'r') as f:
            for i, line in enumerate(f):
                if '-' in line:
                    if len(session) > 1:
                        self._num_events += len(session)
                        if len(session) < self._max_length + 1:
                            for _ in range(self._max_length - len(session) + 1):
                                session.append([0] * 5)
                        self._data.append(session)
                    session = []
                else:
                    session.append([int(j) for j in line.strip().split(',')])

        self._data = np.array(self._data, dtype=np.int32)
        self._num_events_eval = self._num_events - len(self._data)
        self._num_batch = int(float(len(self._data) - 1) / self._batch_size) + 1

        print('Num sessions: ', len(self._data))
        print('Num events: ', self._num_events)

    def next_epoch(self, shuffle=False):
        if shuffle:
            np.random.shuffle(self._data)
        self._batch_index = 0

    def next_batch(self):
        start_idx = self._batch_index * self._batch_size
        end_idx = start_idx + self._batch_size
        self._batch_index += 1
        if self._batch_index == self._num_batch:
            self._batch_index = -1
        return self._data[start_idx: end_idx]

    def has_next(self):
        return self._batch_index != -1
