import os
import argparse
from datetime import datetime
from tqdm import tqdm


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--output_dir')
    parser.add_argument('--sep', default='\t')
    parser.add_argument('--skip_first', type=bool, default=False)
    parser.add_argument('--op', choices=('sample', 'filter'), default='sample')
    parser.add_argument('--year', type=int, default=None)
    parser.add_argument('--fraction', type=float, default=0.1)

    return parser.parse_args()


def get_datetime_from_str(time_stamp_str, time_format):
    dt = datetime.strftime(time_stamp_str, time_format)
    return dt


def filter_event_by_year(path, sep, year, output_dir,
                         skip_first=False, pt=-1,
                         time_format='%Y-%m-%dT%H:%M:%S%Z'):
    file_name = os.path.split(path)[1]
    with open(os.path.join(output_dir, 'filtered-' + file_name), 'w') as wf:
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


def sample_data(path, output_dir, skip_first=False, fraction=0.1):
    file_name = os.path.split(path)[1]
    with open(os.path.join(output_dir, 'filtered-' + file_name), 'w') as wf:
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


if __name__ == '__main__':
    args = _parse_args()
    if args.op == 'sample':
        sample_data(args.path, args.output_dir, args.skip_first, args.fraction)
    else:
        filter_event_by_year(args.path, args.sep, args.year, args.output_dir)
