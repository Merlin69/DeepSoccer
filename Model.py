import tensorflow as tf
import copy
import ToolBox
import Costs


class Model(object):
    def __init__(self, data):

        # Define constants
        self.nb_times = data.nb_saisons
        self.nb_teams = data.nb_teams

        self.data = data

        # Define training and testing set
        self.train_data = self.data.tf_slices['train']
        self.test_data = self.data.tf_slices['test']

        # Define child variablesq
        self.costs = {}
        self.regulizer = {}
        self.train_step = {}
        self.session = None
        self.res = {}
        self.score = {}
        self.dictparam = {}

    def finish_init(self):
        print 'Model created. ' + str(ToolBox.nb_tf_op()) + ' nodes in tf.Graph'

        # Create the session
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def logloss(self, slices, prediction, feature, feature_size, name=None):
        if name is None:
            name = 'll' + '_' + feature
        cost = Costs.Cost(self, name)
        cost.define_logloss(slices, prediction, feature, feature_size)
        return cost

    def define_cost(self, cost, name=None, trainable=False):
        if name is None:
            name = cost.name
        self.costs[name] = cost
        if trainable:
            self.add_opti(name, cost.costs)

    def add_opti(self, name, slices):
        if name not in self.train_step:
            self.train_step[name] = {}
        for s in slices:
            self.train_step[name][s] = tf.train.AdamOptimizer(0.01).minimize(self.costs[name].get(s))

    def set_params(self, param):
        self.dictparam = {}
        for key in param:
            self.dictparam[self.param[key]] = param[key]

    def reset(self):
        self.session.run(tf.initialize_all_variables())

    def train(self, cost, s):
        self.run(self.train_step[cost].get(s))

    def get_cost(self, cost, s):
        return self.run(self.costs[cost].get(s))

    def shuffle(self):
        self.session.run(self.data.shuffle)

    def run(self, x):
        return self.session.run(x, feed_dict=self.dictparam)

    def close(self):
        self.session.close()


