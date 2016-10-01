import tensorflow as tf
import ToolBox
import Model as M
import math

class Elostd(M.Model):
    def __init__(self, train_set, test_set, dictparam):
        super(Elostd, self).__init__(train_set, test_set, dictparam)

        # Define parameters
        self.elo = [range(self.nb_teams)] * self.nb_times
        for t in range(self.nb_times):
            for e in range(self.nb_teams):
                self.elo[t][e] = tf.Variable(0.)

        # Reshape metaparams
        self.param_ = {}
        for key in self.param:
            if key != 'bais_ext':
                self.param_[key] = tf.exp(self.param[key])
            else:
                self.param_[key] = self.param[key]

        # Define the model
        for key, proxy in [('train', self.train_uncroped), ('test', self.test_uncroped)]:
            stack = []
            while proxy['time2']:
                t = proxy['time2'].pop()
                eh = proxy['team_h2'].pop()
                ea = proxy['team_a2'].pop()
                elo_h = self.elo[t][eh]
                elo_a = self.elo[t][ea]
                elomatch = elo_h - elo_a + self.param_['bais_ext']
                elomatch_win = elomatch - self.param_['draw_elo']
                elomatch_los = elomatch + self.param_['draw_elo']
                p_win = tf.inv(1. + tf.exp(-elomatch_win))
                p_los = 1. - tf.inv(1. + tf.exp(-elomatch_los))
                p_tie = 1. - p_los - p_win
                stack.append(tf.pack([p_win, p_tie, p_los]))
            self.res[key] = tf.pack(stack)

        # Define the costs
        self.init_cost()
        for key in ['train', 'test']:
            costs = []
            print 'hiha'
            for e in range(self.nb_teams):
                costs.append(self.param_['metaparam1'] * self.elo[0][e] ** 2)
                for t in range(self.nb_times):
                    costs.append(self.param_['metaparam0'] * self.elo[t][e] ** 2)
            self.regulizer[key].append(tf.add_n(costs))
            print 'yo'
            costs = []
            for e in range(self.nb_teams):
                for t in range(self.nb_times - 1):
                    costs.append(self.param_['metaparam2'] * (self.elo[t][e] - self.elo[t+1][e]) ** 2)
            self.regulizer[key].append(tf.add_n(costs))
        print 'ALLO??'
        # Finish the initialization
        super(Elostd, self).finish_init()

    def get_elos(self):
        return self.session.run(self.elo)

    def reset_elo(self):
        tf.assign(self.elo, tf.zeros([self.nb_teams, self.nb_times]))


