import sys
sys.path.append('../..')

import os

from google_images_download import google_images_download
from tqdm import tqdm

from src.utils.qpath import PROCESSED_DATA_DIR

SAVE_PATH = '/home/nero/NetBeansProjects/ResysDemo/src/main/webapp/img/'
# SAVE_PATH = '/home/nero/Downloads/img/'
KEYWORD_FILE = os.path.join(PROCESSED_DATA_DIR, 'keywords.txt')
response = google_images_download.googleimagesdownload()
arguments = {
    'keywords': '',
    # 'keywords_from_file': KEYWORD_FILE,
    'limit': 1,
    'format': 'jpg',
    'output_directory': SAVE_PATH,
    'no_directory': True,
    'no_numbering': True,
    # 'aspect_ratio': 'square',
}


def download_from_item_map():
    with open(os.path.join(PROCESSED_DATA_DIR, 'new_item_map.txt')) as f:
        for line in tqdm(f):
            song_id, song_artist = line.split('||||')
            song, artist = song_artist.split('-bl-')
            image_path = SAVE_PATH + 'i' + song_id + '.jpg'
            if os.path.exists(image_path):
                continue

            try:
                arguments['keywords'] = song + ' by ' + artist
                absolute_image_paths = response.download(arguments)
                os.rename(
                    list(absolute_image_paths.values())[0][0], image_path)
            except Exception as e:
                continue


def download_from_file():
    response.download(arguments)


if __name__ == '__main__':
    download_from_item_map()
