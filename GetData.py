import csv
import datetime
import ToolBox
import random
import tensorflow as tf
import sys

ALL_FEATURES = ['time', 'team_h', 'team_a', 'res', 'score_h', 'score_a']
DATA_TYPE = tf.float32


class Data:
    def __init__(self, filename, features=ALL_FEATURES, remove_features=[]):

        self.filename = filename
        self.features = [x for x in features if x not in remove_features]

        self.country_to_id = {}
        self.id_to_country = []
        self.nb_teams = 159
        self.nb_saisons = 14
        self.max_daypersaison = 38
        self.nb_matchs = 0

        self.datas = {}
        for key in features:
            self.datas[key] = []

        self.tf_shuffled_datas = {}
        self.tf_datas = {}
        self.tf_slices = {}
        self.shuffle = []

        with open(self.filename, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if row:
                    [score1, score2, year, day, id1, id2] = row
                    self.nb_matchs += 1
                    self.append_match(self.datas, int(id1), int(id2), int(score1), int(score2), int(year) - 2003)

        seed = random.randint(0, sys.maxint)
        for key in self.features:
            self.tf_datas[key] = tf.constant(self.datas[key], dtype=DATA_TYPE)  # Essayer avec variable trainable=False
            self.tf_shuffled_datas[key] = tf.random_shuffle(self.tf_datas[key], seed=seed)

        print 'Data reader created. ' + str(ToolBox.nb_tf_op()) + ' nodes in tf.Graph'

    def add_slices(self, p=None):
        if p is not None:
            slices = [{'left': 0, 'right': int(self.nb_matchs * p), 'name': 'train', 'shuffled': True},
                      {'left': int(self.nb_matchs * p), 'right': -1, 'name': 'test', 'shuffled': True}]
            for s in slices:
                self.create_tf_slice(s)
        print 'Slices created. ' + str(ToolBox.nb_tf_op()) + ' nodes in tf.Graph'

    def create_tf_slice(self, s):
        self.tf_slices[s['name']] = {}
        for key in self.features:
            if s['shuffled']:
                self.tf_slices[s['name']][key] = tf.Variable(0., dtype=DATA_TYPE, validate_shape=False, trainable=False, collections=[])
                shuffled_slice = tf.slice(self.tf_shuffled_datas[key], [s['left'], 0], [s['right'], -1])
                self.shuffle.append(tf.assign(self.tf_slices[s['name']][key], shuffled_slice, validate_shape=False))
            else:
                value = tf.slice(self.tf_datas[key], [s['left'], 0], [s['right'], -1])
                self.tf_slices[s['name']][key] = tf.constant(value, dtype=DATA_TYPE)

    def append_match(self, proxy, id1, id2, score1, score2, date):
        proxy['time'].append(ToolBox.make_vector(date, self.nb_saisons))
        proxy['team_h'].append(ToolBox.make_vector(id1, self.nb_teams))
        proxy['team_a'].append(ToolBox.make_vector(id2, self.nb_teams))
        score_team_h = min(int(score1), 9)
        score_team_a = min(int(score2), 9)
        proxy['score_h'].append(ToolBox.make_vector(score_team_h, 10))
        proxy['score_a'].append(ToolBox.make_vector(score_team_a, 10))
        proxy['res'].append(ToolBox.result_vect(int(score1) - int(score2)))

    def get_elos(self, countries=None, times='all'):
        if countries is None:
            countries = [self.id_to_country[i] for i in range(self.nb_teams)]
        else:
            countries = map(ToolBox.format_name, countries)
        if times == 'all':
            times = range(self.nb_saisons)
        elif times == 'last':
            times = [self.nb_saisons - 1]
        elos = {}
        for country in countries:
            elos[country] = [200*self.elo[self.country_to_id[country]][t] for t in times]
        return elos
