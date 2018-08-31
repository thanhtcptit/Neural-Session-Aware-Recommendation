import tensorflow as tf


class BaseEval:
    def __init__(self, sess, model, config, data_loader, logger=None, init=False):
        """
        Constructing the evaluation
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

        if init:
            # Initialize all variables of the graph
            print('Initialing graph')
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)

        # Init saver
        self.saver = tf.train.Saver()

    def load(self, path):
        raise NotImplementedError

    def run_evaluation(self):
        raise NotImplementedError

    def eval_step(self):
        """
        implement the logic of the train step

        - run the tensorflow session
        :return: any metrics you need to summarize
        """
        raise NotImplementedError
