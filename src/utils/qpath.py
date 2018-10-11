import os
HOME_DIR = os.path.expanduser('~') + '/'
PROJECT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
EXTERNAL_DATA_DIR = os.path.join(PROJECT_DIR, 'data/external/')
INTERIM_DATA_DIR = os.path.join(PROJECT_DIR, 'data/interim/')
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data/processed/')
RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'data/raw/')
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'models/')
SUMMARY_DIR = os.path.join(PROJECT_DIR, 'summary_dir/')
MODELS_SRC = os.path.join(PROJECT_DIR, 'src/models')
UTILS = os.path.join(PROJECT_DIR, 'src/utils/')