import GetData
import Elostd
import Elosplit
import time
import sys
import MetaOpti
import csv
from scipy.optimize import differential_evolution
import numpy as np
import math
import random
import tensorflow as tf


data = GetData.Data('match_light.txt')
data.add_slices(0.5)
model = Elostd.Elostd(data)

paramStd = {'metaparam0': -10.,
            'metaparam1': -10.,
            'metaparam2': 2.,
            'bais_ext': 0.5,
            'draw_elo': -0.36}

model.set_params(paramStd)
model.shuffle()
model.reset()
for i in xrange(200):
    model.train_res('train')
print model.get_cost('train'), model.get_cost('test')

model.close()
sys.exit()

