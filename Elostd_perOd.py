import tensorflow as tf
import ToolBox
import Model as M
import math

class Elostd(M.Model):
    def __init__(self, train_set, test_set, dictparam):
        super(Elostd, self).__init__(train_set, test_set, dictparam)

        # Define parameters
        self.elo = tf.Variable(tf.zeros([self.nb_teams, self.nb_times]))

        # Reshape metaparams
        self.param_ = {}
        for key in self.param:
            if key != 'bais_ext':
                self.param_[key] = tf.exp(self.param[key])
            else:
                self.param_[key] = self.param[key]

        # Define the model
        for key, proxy in [('train', self.train_data), ('test', self.test_data)]:
            elomatch = ToolBox.get_elomatch(proxy['team_h'] - proxy['team_a'], proxy['time'], self.elo)
            elomatch += self.param_['bais_ext']
            elomatch_win = elomatch - self.param_['draw_elo']
            elomatch_los = elomatch + self.param_['draw_elo']
            p_win = tf.inv(1. + tf.exp(-elomatch_win))
            p_los = 1. - tf.inv(1. + tf.exp(-elomatch_los))
            p_tie = 1. - p_los - p_win
            self.res[key] = tf.pack([p_win, p_tie, p_los], axis=1)

        # Define the costs
        self.init_cost()
        for key in ['train', 'test']:
            cost = ToolBox.get_raw_elo_cost(self.param_['metaparam0'], self.param_['metaparam1'], self.elo, self.nb_times)
            self.regulizer[key].append(cost)

            cost = ToolBox.get_timediff_elo_cost(self.param_['metaparam2'], self.elo, self.nb_times)
            self.regulizer[key].append(cost)

        # Finish the initialization
        super(Elostd, self).finish_init()


    def get_elos(self):
        return self.session.run(self.elo)

    def reset_elo(self):
        tf.assign(self.elo, tf.zeros([self.nb_teams, self.nb_times]))


