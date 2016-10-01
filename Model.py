import tensorflow as tf
import copy
import ToolBox


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
        self.cost_entropy_res = {}
        self.cost_regularized_res = {}
        self.cost_entropy_score = {}
        self.cost_regularized_score = {}
        self.train_step = {}
        self.session = None
        self.res = {}
        self.score = {}
        self.dictparam = {}

    def init_cost(self):
        for key, proxy in [('train', self.train_data), ('test', self.test_data)]:
            if key in self.res:
                entropies_res = tf.log(tf.reduce_sum(proxy['res'] * self.res[key], reduction_indices=[1]) + 1e-9)
                self.cost_entropy_res[key] = tf.reduce_mean(-entropies_res)
            if key in self.score:
                pr = tf.squeeze(tf.batch_matmul(tf.expand_dims(proxy['score_h'], 1), self.score[key]), [1])
                entropies_score = tf.log(tf.reduce_sum(pr * proxy['score_a'], reduction_indices=[1]) + 1e-9)
                self.cost_entropy_score[key] = tf.reduce_mean(-entropies_score)

            self.regulizer[key] = []

    def finish_init(self):
        # Define the regularized cost
        for key in ['train', 'test']:

            if key in self.cost_entropy_res:
                costs_res = copy.copy(self.regulizer[key])
                costs_res.append(self.cost_entropy_res[key])
                self.cost_regularized_res[key] = tf.add_n(costs_res)

            if key in self.cost_entropy_score:
                costs_score = copy.copy(self.regulizer[key])
                costs_score.append(self.cost_entropy_score[key])
                self.cost_regularized_score[key] = tf.add_n(costs_score)

        print 'Model created. ' + str(ToolBox.nb_tf_op()) + ' nodes in tf.Graph'

        # Create the session
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    # def add_cost(self, cost, trainable=True):

    def add_optimizer(self, s):
        self.train_step[s] = tf.train.AdamOptimizer(0.01).minimize(self.cost_regularized_res[s])

    def set_params(self, param):
        self.dictparam = {}
        for key in param:
            self.dictparam[self.param[key]] = param[key]

    def reset(self):
        self.session.run(tf.initialize_all_variables())

    def train_res(self, s):
        self.run(self.train_step[s])

    def get_cost(self, s):
        return self.session.run(self.cost_entropy_res[s], feed_dict=self.dictparam)

    def shuffle(self):
        self.session.run(self.data.shuffle)

    def run(self, x):
        self.session.run(x, feed_dict=self.dictparam)

    def close(self):
        self.session.close()
