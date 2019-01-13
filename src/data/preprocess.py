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


def extract_time_context_utc(utc):
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


def extract_time_context_raw(timestamp, ts_format='%Y-%m-%d %H:%M:%S'):
    dt = datetime.strptime(timestamp, ts_format)
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
    return week_day, half_month_ped


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=RAW_DATA_DIR + 'lastfm.tsv',
                        help='Path to the dataset')
    parser.add_argument('--max_valid_seq_len', type=int, default=500,
                        help='The maximum length allow per session')
    parser.add_argument('--max_session_len', type=int, default=10)
    parser.add_argument('--min_session_len', type=int, default=2)
    parser.add_argument('--min_occur', type=int, default=10)
    parser.add_argument('--min_session_per_user', type=int, default=5)
    parser.add_argument('--time_interval', type=int, default=3600)
    parser.add_argument('--pu', type=int, default=0,
                        help='Position of user id')
    parser.add_argument('--pi', type=int, default=5,
                        help='Position of item id')
    parser.add_argument('--pt', type=int, default=1,
                        help='Position of timestamp')
    parser.add_argument('--time_format', type=str,
                        default='%Y-%m-%dT%H:%M:%S%Z')
    parser.add_argument('--skip_first', type=bool,
                        default=False)
    parser.add_argument('--sep', type=str, default='\t')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--op', choices=['split', 'all'],
                        help='''
                        [all] Preprocess data + Split session
                        [split] Split sessions
                        (assume the preprocess step had complete)
                        ''')
    return parser.parse_args()


def parse_data(args):
    with open(args.path) as f:
        if args.skip_first:
            f.readline()
        for i, line in tqdm(enumerate(f)):
            line_data = line.strip().split(args.sep)
            try:
                usr, item, ts = \
                    line_data[args.pu], line_data[args.pi], line_data[args.pt]
                if args.time_format is not None:
                    ts = date2utc(ts, args.time_format)
                yield (usr, item, ts)
            except IndexError:
                print("IndexError: list index out of range for line {} ('{}'),"
                      " ignoring".format(i, line.strip()))
                continue


def preprocess(args, stream):
    data = list()
    occurrences = collections.defaultdict(lambda: 0)
    # Read from stream
    for user, item, ts in stream:
        data.append((user, item, ts))
        occurrences[item] += 1
    seq_dict = collections.defaultdict(list)
    user_item_dict = collections.defaultdict(list)

    # Remove items that occurred infrequently
    dropped_events = 0
    for user, item, ts in tqdm(data):
        if occurrences[item] < args.min_occur:
            dropped_events += 1
            continue
        seq_dict[user].append([ts, item])
        user_item_dict[user].append(item)

    seq2_dict = collections.defaultdict(list)
    for user, seq in seq_dict.items():
        seq = [(ts, item) for ts, item in sorted(seq)]
        seq2_dict[user] = seq

    # Create user vocab and item vocab
    items = set(itertools.chain(*user_item_dict.values()))
    users = set(seq2_dict.keys())
    print('First filter (item occur < {}): '.format(args.min_occur))
    print('- Num users: ', len(users))
    print('- Num items: ', len(items))
    print('- Num events: ', len(data) - dropped_events)

    item2id = dict(zip(items, range(1, len(items) + 1)))
    user2id = dict(zip(users, range(1, len(users) + 1)))

    user_data_dir = os.path.join(
        PROCESSED_DATA_DIR, '{}user_dir{}'.format(args.prefix, args.suffix))
    if os.path.exists(user_data_dir):
        shutil.rmtree(user_data_dir)
    os.makedirs(user_data_dir)
    for user in tqdm(seq2_dict.keys()):
        user_file = os.path.join(user_data_dir, str(user))
        with open(user_file, 'w') as uf:
            for iid in seq2_dict[user]:
                uf.write("{},{},{}\n".format(
                    user2id[user], item2id[iid[1]], iid[0]))
            uf.write('EOF')


def cutting(args, origin_session):
    sessions = []
    is_val_session = 0
    events_count = 0
    if 1 < len(origin_session) <= args.max_valid_seq_len:
        is_val_session = 1
        events_count = len(origin_session)
        ns = int(float(len(origin_session) - 1) / args.max_session_len) + 1
        for s in range(ns):
            start = s * args.max_session_len
            end = start + args.max_session_len + 1
            if len(origin_session[start:end]) >= args.min_session_len:
                sessions.append(origin_session[start:end])
    return sessions, is_val_session, events_count


def split_session(args):
    num_origin_sessions = 0
    num_cut_sessions = 0
    num_events = 0
    num_users = 0

    user_data_dir = os.path.join(
        PROCESSED_DATA_DIR, '{}user_dir{}'.format(args.prefix, args.suffix))
    if os.path.exists(
            PROCESSED_DATA_DIR + '{}train{}'.format(args.prefix, args.suffix)):
        try:
            os.remove(PROCESSED_DATA_DIR + '{}train{}'.format(
                args.prefix, args.suffix))
            os.remove(PROCESSED_DATA_DIR + '{}test{}'.format(
                args.prefix, args.suffix))
            os.remove(PROCESSED_DATA_DIR + '{}dev{}'.format(
                args.prefix, args.suffix))
        except OSError:
            pass
    for i, file in tqdm(enumerate(sorted(os.listdir(user_data_dir)))):
        sessions = []
        session = []
        last_id = -1
        num_sessions = 0
        user_events = 0
        user_data_file = os.path.join(user_data_dir, file)
        with open(user_data_file, 'r') as f:
            for line in f:
                if line == 'EOF':
                    cut_session, is_val_session, events_count = \
                        cutting(args, session)
                    sessions.extend(cut_session)
                    num_sessions += is_val_session
                    user_events += events_count
                    break

                data = line.strip().split(',')
                if len(session) == 0:
                    session.append(data)
                    last_id = data[1]
                elif math.fabs(float(data[2]) - float(session[-1][2])) \
                        < args.time_interval:
                    if last_id == data[1]:
                        continue
                    last_id = data[1]
                    session.append(data)
                else:
                    cut_session, is_val_session, events_count = \
                        cutting(args, session)
                    sessions.extend(cut_session)
                    num_sessions += is_val_session
                    user_events += events_count
                    session = [data]
                    last_id = data[1]

        if num_sessions >= args.min_session_per_user:
            num_origin_sessions += num_sessions
            num_cut_sessions += len(sessions)
            num_users += 1
            num_events += user_events
            save_user_session(args, sessions)

    print('Second filter ' +
          '(session length >= {} and sessions per user > {}): '.format(
            args.min_session_len, args.min_session_per_user))
    print('- Total origin sessions', num_origin_sessions)
    print('- Total cut sessions: ', num_cut_sessions)
    print('- Event per origin sessions ',
          float(num_events) / num_origin_sessions)
    print('- Event per cut sessions ', float(num_events) / num_cut_sessions)
    print('- Total events: ', num_events)
    print('- Average sessions length: ',
          float(num_events) / num_origin_sessions)
    print('- Sessions per user: ', float(num_origin_sessions) / num_users)


def save_user_session(args, sessions):
    if len(sessions) <= 20:
        train_idx = len(sessions) - 2
        dev_idx = train_idx + 1
    else:
        train_idx = int(len(sessions) * 0.9)
        dev_idx = int(train_idx + len(sessions) * 0.05)

    with open(PROCESSED_DATA_DIR + '{}train{}'.format(
            args.prefix, args.suffix), 'a') as f1:
        for sess in sessions[:train_idx]:
            for s in sess:
                h, d, m = extract_time_context_utc(s[2])
                f1.write('{},{},{},{},{}\n'.format(s[0], s[1], h, d, m))
            f1.write('-----\n')
    with open(PROCESSED_DATA_DIR + '{}dev{}'.format(
            args.prefix, args.suffix), 'a') as f1:
        for sess in sessions[train_idx:dev_idx]:
            for s in sess:
                h, d, m = extract_time_context_utc(s[2])
                f1.write('{},{},{},{},{}\n'.format(s[0], s[1], h, d, m))
            f1.write('-----\n')
    with open(PROCESSED_DATA_DIR + '{}test{}'.format(
            args.prefix, args.suffix), 'a') as f1:
        for sess in sessions[dev_idx:]:
            for s in sess:
                h, d, m = extract_time_context_utc(s[2])
                f1.write('{},{},{},{},{}\n'.format(s[0], s[1], h, d, m))
            f1.write('-----\n')


def clean_data(path, file, train_items, users_map, items_map):
    new_data = []
    with open(path + file, 'r') as f:
        for line in f:
            if '-' in line:
                new_data.append(line)
                continue
            data = line.strip().split(',')
            if train_items[int(data[1])] == 0:
                continue
            new_data.append(data)

    event_count = 0
    with open(path + 'clean-' + file, 'w') as f:
        for data in new_data:
            if '-' in data:
                f.write(data)
                continue
            event_count += 1
            f.write('{},{},{},{},{}\n'.format(users_map[int(data[0])],
                                              items_map[int(data[1])],
                                              data[2], data[3], data[4]))
    print('- clean-{}: {} events'.format(file, event_count))
    os.remove(path + file)


def remove_unseen_data(args):
    users = set()
    items = set()
    train_items = collections.defaultdict(lambda: 0)

    with open(PROCESSED_DATA_DIR + '{}train{}'.format(
            args.prefix, args.suffix), 'r') as f:
        for line in f:
            if '-' in line:
                continue
            data = line.strip().split(',')
            users.add(int(data[0]))
            items.add(int(data[1]))
            train_items[int(data[1])] = 1

    with open(PROCESSED_DATA_DIR + '{}test{}'.format(
            args.prefix, args.suffix), 'r') as f:
        for line in f:
            if '-' in line:
                continue
            data = line.strip().split(',')
            if train_items[int(data[1])] == 0:
                continue
            users.add(int(data[0]))

    with open(PROCESSED_DATA_DIR + '{}dev{}'.format(
            args.prefix, args.suffix), 'r') as f:
        for line in f:
            if '-' in line:
                continue
            data = line.strip().split(',')
            if train_items[int(data[1])] == 0:
                continue
            users.add(int(data[0]))

    users_map = dict(zip(users, range(1, len(users) + 1)))
    items_map = dict(zip(items, range(1, len(items) + 1)))
    print('Third filter (remove item not exist in the train set): ')
    print('- Num users: ', len(users))
    print('- Num items: ', len(items))

    clean_data(PROCESSED_DATA_DIR, '{}train{}'.format(
        args.prefix, args.suffix),
        train_items, users_map, items_map)
    clean_data(PROCESSED_DATA_DIR, '{}test{}'.format(
        args.prefix, args.suffix),
        train_items, users_map, items_map)
    clean_data(PROCESSED_DATA_DIR, '{}dev{}'.format(
        args.prefix, args.suffix),
        train_items, users_map, items_map)
    with open(PROCESSED_DATA_DIR +
              'clean-{}train{}-metadata'.format(
                args.prefix, args.suffix), 'w') as f:
        f.write(str(len(items)) + '\n')
        f.write(str(len(users)) + '\n')
        f.write(str(args.max_session_len))


if __name__ == '__main__':
    args = _parse_args()
    # Preprocess data & create train - val - test
    if args.op == 'all':
        stream = parse_data(args)
        preprocess(args, stream)

    split_session(args)
    remove_unseen_data(args)
