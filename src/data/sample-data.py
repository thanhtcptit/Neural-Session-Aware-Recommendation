import os
import argparse
import collections
from datetime import datetime
from tqdm import tqdm


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default='/home/ntq/thanhtc/RNN-for-Resys' +
                        '/data/raw/avito/filtered-VisitsStream.txt')
    parser.add_argument('--output_dir',
                        default=None)
    parser.add_argument('--sep', default=',')
    parser.add_argument('--pu', type=int, default=0,
                        help='Position of user id')
    parser.add_argument('--pi', type=int, default=5,
                        help='Position of item id')
    parser.add_argument('--pt', type=int, default=1,
                        help='Position of timestamp')
    parser.add_argument('--skip_first', type=bool, default=False)
    parser.add_argument('--op',
                        choices=('fraction', 'year', 'top_items', 'user_item'),
                        default='fraction')
    parser.add_argument('--values', type=float, default=0.1)

    return parser.parse_args()


def get_datetime_from_str(time_stamp_str, time_format):
    dt = datetime.strptime(time_stamp_str.replace('Z', 'GMT'), time_format)
    return dt


def sample_data_by_year(path, sep, output_dir, year,
                        skip_first=False, pt=1,
                        time_format='%Y-%m-%dT%H:%M:%S%Z'):
    input_dir, file_name = os.path.split(path)
    if output_dir is None:
        output_dir = input_dir

    with open(os.path.join(
            output_dir, str(year) + '-sample-' + file_name), 'w') as wf:
        with open(path, 'r') as rf:
            if skip_first:
                rf.readline()
            for i, line in tqdm(enumerate(rf), desc='Line'):
                line_data = line.strip().split(sep)
                try:
                    ts = line_data[pt]
                    dt = get_datetime_from_str(ts, time_format)
                    if dt.year == year:
                        wf.write(line)
                except IndexError:
                    print("IndexError: list index out of range for line "
                          "{} ('{}'), ignoring".format(i, line.strip()))
                    continue


def sample_data_by_fraction(path, output_dir, skip_first=False, fraction=0.1):
    input_dir, file_name = os.path.split(path)
    if output_dir is None:
        output_dir = input_dir

    with open(os.path.join(
            output_dir, str(fraction) + '-sample-' + file_name), 'w') as wf:
        with open(path, 'r') as rf:
            if skip_first:
                rf.readline()
            file_length = sum(1 for line in rf)
            max_size = int(file_length * fraction)
            print('File length: ', file_length)
            print('Sample size: ', max_size)
            rf.seek(0)
            for i, line in tqdm(enumerate(rf), 'Line'):
                if i == max_size:
                    break
                wf.write(line)


def sample_data_by_items(path, sep, output_dir, top=10000):
    input_dir, file_name = os.path.split(path)
    if output_dir is None:
        output_dir = input_dir

    print('Read ads ID from category 43')
    with open(os.path.join('/home/ntq/thanhtc/RNN-for-Resys' +
                           '/data/raw/avito/Category43Ads.txt')) as f:
        valid_ads_id = set(f.read().split('\n'))
    print('- Total ID: ', len(valid_ads_id))
    print('Count occurences of ads')
    occurs = collections.defaultdict(lambda: 0)
    with open(path) as rf:
        for line in tqdm(rf, 'Events'):
            usr, _, ad, time = line.strip().split(sep)
            if ad in valid_ads_id:
                occurs[ad] += 1
    sorted_occurs = sorted(occurs.items(), key=lambda kv: kv[1])[-top:]
    valid_items = set([i for i, v in sorted_occurs])
    print(len(valid_items))

    with open(os.path.join(
            output_dir, str(top) + '-items-' + file_name), 'w') as wf:
        with open(path) as rf:
            for line in tqdm(rf, 'Events'):
                usr, _, ad, time = line.strip().split('\t')
                if ad in valid_items:
                    wf.write(line)


def get_user_item_from_file(path, output_dir,
                            sep, skip_first=False, pu=0, pi=5):
    input_dir, file_name = os.path.split(path)
    users = set()
    items = set()
    if output_dir is None:
        output_dir = input_dir
    with open(path) as rf:
        with open(os.path.join(
                output_dir, 'ui-' + file_name), 'w') as wf:
            if skip_first:
                rf.readline()
            for line in tqdm(rf, desc='Line'):
                if '-' in line:
                    continue
                data = line.strip().split(sep)
                users.add(data[pu])
                items.add(data[pi])
                wf.write('{},{}\n'.format(data[pu], data[pi]))

    with open(os.path.join(output_dir, 'u-' + file_name), 'w') as f:
        f.write('\n'.join(users))
    with open(os.path.join(output_dir, 'i-' + file_name), 'w') as f:
        f.write('\n'.join(items))


if __name__ == '__main__':
    args = _parse_args()
    if args.op == 'fraction':
        sample_data_by_fraction(args.path, args.output_dir,
                                args.skip_first, args.values)
    elif args.op == 'year':
        sample_data_by_year(args.path, args.sep,
                            args.output_dir, int(args.values), pt=args.pt)
    elif args.op == 'top_items':
        sample_data_by_items(args.path, args.output_dir, int(args.values))
    else:
        get_user_item_from_file(
            args.path, args.output_dir, args.sep, pu=args.pu, pi=args.pi)
