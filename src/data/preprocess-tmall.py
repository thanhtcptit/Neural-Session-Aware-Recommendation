import os
import time
from calendar import timegm
from datetime import datetime

from collections import defaultdict
from tqdm import tqdm

from src.utils.qpath import *


def utc2date(utc):
    return datetime.utcfromtimestamp(float(utc))


tmall_data = os.path.join(RAW_DATA_DIR, 'tmall.csv')
data_count = defaultdict(lambda: 0)

with open(tmall_data) as f:
    with open(os.path.join(RAW_DATA_DIR, 'tmall-sample-7.csv'), 'w') as wf:
        f.readline()
        for line in tqdm(f):
            data = line.strip().split('\t')
            date = utc2date(data[3])
            data_count[date.month] += 1
            if date.month == 7:
                wf.write('{},{},{}\n'.format(data[0], data[1], data[3]))

print(data_count)
