import tensorflow as tf
import ToolBox
import Model as M
import math
import Costs

PARAMS = ['metaparam0', 'metaparam1', 'metaparam2', 'bais_ext', 'draw_elo']


class Elostd(M.Model):
    def __init__(self, data):
        super(Elostd, self).__init__(data)

        # Define parameters
        self.elo = tf.Variable(tf.zeros([self.data.nb_teams, self.data.nb_saisons]))

        self.param = {}
        for key in PARAMS:
            self.param[key] = tf.placeholder(tf.float32)

        # Reshape metaparams
        self.param_ = {}
        for key in self.param:
            if key != 'bais_ext':
                self.param_[key] = tf.exp(self.param[key])
            else:
                self.param_[key] = self.param[key]

        self.elomatch = {}
        # Define the model

        for key in self.data.tf_slices:
            s = self.data.tf_slices[key]
            elomatch = ToolBox.get_elomatch(s['team_h'] - s['team_a'], s['time'], self.elo)
            elomatch += self.param_['bais_ext']
            self.elomatch[key] = elomatch
            elomatch_win = elomatch - self.param_['draw_elo']
            elomatch_los = elomatch + self.param_['draw_elo']
            p_win = tf.inv(1. + tf.exp(-elomatch_win))
            p_los = 1. - tf.inv(1. + tf.exp(-elomatch_los))
            p_tie = 1. - p_los - p_win
            self.res[key] = tf.pack([p_win, p_tie, p_los], axis=1)

        # Define the costs
        regulizer = {}
        for key in self.data.tf_slices:
            regulizer_list = []
            cost = ToolBox.get_raw_elo_cost(self.param_['metaparam0'], self.param_['metaparam1'], self.elo, self.data.nb_saisons)
            regulizer_list.append(cost)

            cost = ToolBox.get_timediff_elo_cost(self.param_['metaparam2'], self.elo, self.data.nb_saisons)
            regulizer_list.append(cost)

            regulizer[key] = tf.add_n(regulizer_list)

        self.regulizer = Costs.Cost(self, 'reg', costs=regulizer)

    def get_elos(self):
        return self.session.run(self.elo)

    def reset_elo(self):
        tf.assign(self.elo, tf.zeros([self.data.nb_teams, self.data.nb_saisons]))


