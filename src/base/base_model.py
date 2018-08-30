import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        self.config = config

    def build_model(self):
        raise NotImplementedError
