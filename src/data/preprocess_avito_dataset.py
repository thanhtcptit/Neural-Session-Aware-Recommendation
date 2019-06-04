import sys
sys.path.append('../..')

import argparse
import collections
import itertools
import math
import os
import shutil
import sys
import time
from calendar import timegm
from datetime import datetime
from tqdm import tqdm
from src.utils.qpath import *


def date2utc(date, ts_format='%Y-%m-%dT%H:%M:%S%Z'):
    return timegm(time.strptime(date.replace('Z', 'GMT'), ts_format))


def utc2date(utc):
    return datetime.utcfromtimestamp(float(utc))


def get_year(utc):
    dt = datetime.utcfromtimestamp(float(utc))
    return dt.year


def get_month(utc):
    dt = datetime.utcfromtimestamp(float(utc))
    return dt.month


def get_day(utc):
    dt = datetime.utcfromtimestamp(float(utc))
    return dt.day


def extract_time_context(utc):
    dt = datetime.utcfromtimestamp(float(utc))
    hour = dt.hour
    month = dt.month
    week_day = dt.weekday()
    if month == 12:
        day_of_month = datetime(day=1, month=1, year=dt.year + 1) \
                       - datetime(day=1, month=month, year=dt.year)
    else:
        day_of_month = datetime(day=1, month=month + 1, year=dt.year) \
                       - datetime(day=1, month=month, year=dt.year)
    day_of_month = day_of_month.days
    if dt.day < day_of_month / 2:
        half_month_ped = month * 2 - 1
    else:
        half_month_ped = month * 2
    return hour, week_day, half_month_ped


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ads_info_path',
        default='/home/ntq/thanhtc/data/avito/AdsInfo.tsv')
    parser.add_argument(
        '--search_stream_path',
        default='/home/ntq/thanhtc/data/avito/trainSearchStream.tsv')
    parser.add_argument(
        '--search_info_path',
        default='/home/ntq/thanhtc/data/avito/SearchInfo.tsv')
    parser.add_argument(
        '--visit_stream_path',
        default='/home/ntq/thanhtc/data/avito/VisitsStream.tsv')
    parser.add_argument(
        '--output_path',
        default='/home/ntq/thanhtc/RNN-for-Resys/data/raw/avito')
    return parser.parse_args()


def filter_ad_by_category(ads_info_path, output_path):
    data = list()
    print('Filter ads outside category 43: ')
    with open(ads_info_path) as f:
        f.readline()
        for i, line in tqdm(enumerate(f), 'Ads'):
            line_data = line.strip().split('\t')
            ad_id, cat = line_data[0], line_data[2]
            if cat == '43':
                data.append(ad_id)

    print('- Total valid Ads: ', len(data))
    with open(os.path.join(output_path, 'Category43Ads.txt'), 'w') as f:
        f.write('\n'.join(set(data)))


def filter_search_by_user(search_info_path, output_path):
    with open(os.path.join(output_path, 'users.txt')) as f:
        users = set(f.read().split('\n'))

    valid_search_count = 0
    usr_list = set()
    print('Filter search event by users: ')
    with open(os.path.join(output_path, 'SearchStream.txt'), 'w') as wf:
        with open(search_info_path) as f:
            f.readline()
            for i, line in tqdm(enumerate(f), 'Searchs'):
                line_data = line.strip().split('\t')
                search_id, time, usr = line_data[0], line_data[1], line_data[3]
                if usr in users:
                    usr_list.add(usr)
                    wf.write('{},{},{}\n'.format(search_id, time, usr))
                    valid_search_count += 1

    print('- Total valid search: ', valid_search_count)
    print(len(usr_list))


def filter_events_by_id(visit_stream_path, output_path):
    print('Read ads ID from category 43')
    with open(os.path.join(output_path, 'Category43Ads.txt')) as f:
        valid_ads_id = set(f.read().split('\n'))
    print('- Total ID: ', len(valid_ads_id))

    print('Count occurences of ads')
    occurs = collections.defaultdict(lambda: 0)
    with open(visit_stream_path) as rf:
            rf.readline()
            for line in tqdm(rf, 'Events'):
                usr, _, ad, time = line.strip().split('\t')
                if ad in valid_ads_id:
                    occurs[ad] += 1
    print('- Number of ad has more than 20 click: ',
          sum([1 for i in occurs if occurs[i] >= 20]))

    print('Filter ad outside category 43 and has less than 20 click')
    valid_events_count = 0
    users = set()
    ads = set()
    with open(os.path.join(
            output_path, 'filtered-VisitsStream.txt'), 'w') as wf:
        with open(visit_stream_path) as rf:
            rf.readline()
            for i, line in enumerate(rf):
                if i % 1000000 == 0:
                    print('- Iter {}: {} events'.format(
                        i, valid_events_count), end='\r')
                usr, _, ad, time = line.strip().split('\t')
                if ad in valid_ads_id and occurs[ad] >= 20:
                    valid_events_count += 1
                    users.add(usr)
                    ads.add(ad)
                    wf.write(line)

    with open(os.path.join(output_path, 'users.txt'), 'w') as f:
        f.write('\n'.join(users))

    with open(os.path.join(output_path, 'ads.txt'), 'w') as f:
        f.write('\n'.join(ads))

    print('- Total valid events: ', valid_events_count)
    print('- Total users: ', len(users))
    print('- Total ads: ', len(ads))


def write_session(usr, session, file):
    if len(session) < 4:
        return
    with open(os.path.join(PROCESSED_DATA_DIR, file), 'a') as f:
        for ad, time in session:
            h, d, m = extract_time_context(time)
            f.write('{},{},{},{},{}\n'.format(usr, ad, h, d, m))


def split_session(output_path):
    files = ['test', 'dev', 'train']
    data = collections.defaultdict(list)
    usr_sess_count = collections.defaultdict(lambda: 0)
    with open(os.path.join(
            output_path, 'filtered-VisitsStream.txt')) as f:
        for line in f:
            if line in ['\n', '\r\n']:
                continue
            event = line.strip().split(',')
            usr, ad, time = event
            if data[usr]:
                if math.fabs(float(data[usr][-1][1]) - float(time)) < 3600:
                    data[usr].append((ad, time))
                else:
                    i = 2 if usr_sess_count[usr] > 2 else usr_sess_count[usr]
                    write_session(usr, data[usr], files[i])
                    usr_sess_count[usr] += 1
                    data[usr] = [(ad, item)]
            else:
                data[usr].append((ad, time))


if __name__ == '__main__':
    args = _parse_args()
    # filter_ad_by_category(args.ads_info_path, args.output_path)
    # filter_search_by_user(args.search_info_path, args.output_path)
    filter_events_by_id(args.visit_stream_path, args.output_path)
