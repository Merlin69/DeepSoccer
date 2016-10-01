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
import ToolBox

t = time.time()
data = GetData.Data('match_light.txt')
data.add_slices(0.5)

print 'data', time.time() - t
t = time.time()

model = Elostd.Elostd(data)

ll_res = model.logloss(['train', 'test'], model.res, 'res', 1, name='ll_res')
model.define_cost(ll_res, trainable=False)
model.define_cost(ll_res + model.regulizer, trainable=True, name='rll_res')
model.finish_init()


print 'model', time.time() - t
t = time.time()

paramStd = {'metaparam0': -10.,
            'metaparam1': -10.,
            'metaparam2': 2.,
            'bais_ext': 0.5,
            'draw_elo': -0.36}
model.set_params(paramStd)
model.shuffle()
model.reset()

print 'reset', time.time() - t
t = time.time()

for i in xrange(60):
    model.train('rll_res', 'train')

print 'loop1', time.time() - t
t = time.time()

for i in xrange(60):
    model.train('rll_res', 'train')

print 'loop2', time.time() - t
t = time.time()

temp = model.get_cost('ll_res', 'train'), model.get_cost('ll_res', 'test')

print 'get_cost', time.time() - t

sys.exit()
