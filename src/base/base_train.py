import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, config, data_loader, logger=None):
        """
        Constructing the trainer
        :param sess: TF.Session() instance
        :param model: The model instance
        :param config: config namespace which will contain all the configurations you have specified in the json
        :param logger: logger class which will summarize and write the values to the tensorboard
        :param data_loader: The data loader if specified. (You will find Dataset API example)
        """
        # Assign all class attributes
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data_loader = data_loader

        # Init saver
        self.saver = tf.train.Saver()

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def run_training(self):
        raise NotImplementedError

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary

        :param epoch: take the number of epoch if you are interested
        :return:
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step

        - run the tensorflow session
        :return: any metrics you need to summarize
        """
        raise NotImplementedError
