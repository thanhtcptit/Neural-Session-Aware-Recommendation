import sys
import os

from tqdm import tqdm
from collections import defaultdict

from src.utils.qpath import RAW_DATA_DIR
from src.data.preprocess import date2utc, utc2date, get_day, get_month


def events_distributed(file_name, tformat='%Y-%m-%dT%H:%M:%S%Z', pt=1, de=','):
    days = defaultdict(lambda: 0)
    months = defaultdict(lambda: 0)
    with open(os.path.join(RAW_DATA_DIR, file_name)) as f:
        for line in tqdm(f):
            ts = line.strip().split(de)[pt]
            if ts is not None:
                ts = date2utc(ts, tformat)
            day = get_day(ts)
            month = get_month(ts)
            months[month] += 1
            days['{}-{}'.format(day, month)] += 1

    print(months)
    print(days)


if __name__ == '__main__':
    file_name = 'avito/5162-items-filtered-VisitsStream.txt'
    tformat = '%Y-%m-%d %H:%M:%S.%f'
    pt = 3
    de = '\t'
    events_distributed(file_name, tformat, pt, de)
