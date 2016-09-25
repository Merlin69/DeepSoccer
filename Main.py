import GetData
import Elostd
import Elosplit
import time
import sys
import MetaOpti
import csv

# Get the datas (50% train, 50% test)
data = GetData.Data('match_light.txt', 60)
data.get(0.5)
print 'dat get'
param = {'metaparam0': 0.1, 'metaparam1': 0.1, 'metaparam2': 500.0, 'draw_elo': 0.4, 'bais_ext': 0.6}
print 'param set'
model = Elostd.Elostd(data.train, data.test, param)
print 'model defined'
param_range = {'metaparam0': (1.e-05, 2., 'exp'),
               'metaparam1': (1.e-05, 2., 'exp'),
               'metaparam2': (10., 3., 'exp'),
               'draw_elo': (0.74, 1.3, 'exp'),
               'bais_ext': (0.53, 0.1, 'lin')}

it84 = {'metaparam0': (3.674509963873269e-06, 1.0100393632706985, 'exp'),
 'metaparam1': (2.2060718676360033e-07, 1.0051276249596388, 'exp'),
 'metaparam2': (5.896819852763622, 1.0622580192504276, 'exp'),
 'bais_ext': (0.49647725989888003, 0.0035184372088832034, 'lin'),
 'draw_elo': (0.696919145701818, 1.0059254041847643, 'exp')}


print 'range defined'
optimizer = MetaOpti.MetaOpti(data, param_range, param, model, lambda: model.get_cost('test'), 5, 0.8)
print 'opti defined'
for i in range(100):
    print 'indice', i
    if not i%5 and i!=0:
        optimizer.data.get(0.5)
        print 'DATA IN THERE'
    optimizer.model.reset_elo()
    print 'PARAMS RESETED'
    optimizer.mongo_opti()
    print optimizer.paramrange


this_time = time.time()
for k in range(100):
    print k, model.get_cost('train'), model.get_cost('test')
    model.train_res()
print "train: "+str(model.get_cost('test'))
print "test: "+str(model.get_cost('train'))
print int(100. * (time.time() - this_time))





sys.exit()

# Fix some meta-parameters
param = {'metaparam0': 0.00015, 'metaparam1': 0.00025, 'metaparam2': 11., 'bais_ext': 0.}

param_range = {
    'metaparam0': [0.05 ** (i - 5) for i in range(10)],
    'metaparam1': [0.05 ** (i - 5) for i in range(10)],
    'metaparam2': [10 * 0.01 ** (i-5) for i in range(10)],
    'bais_ext': [0. + (i-5)*2 for i in range(10)]}





sys.exit()


print param_range

model = Elostd.Elostd(data.train, data.test, param)

print data.nb_teams

for i in range(10):
    print i
    model.train_res()

print model.get_cost('test')

data.set_elos(model.get_elos())
l = data.get_elos(times='last')
print data.nb_teams


optimizer = MetaOpti.MetaOpti(data, param_range, param, model, lambda: model.get_cost('test'))

for i in range(100):
    optimizer.data.get(0.5)
    optimizer.model.reset_param()
    optimizer.mongo_opti()
    print optimizer.param

# Get the datas (50% train, 50% test)
data = GetData.Data('data.txt', 12)
data.get(0.5)

# Fix some meta-parameters
param = {'metaparam0': 0.015, 'metaparam1': 0.025, 'metaparam2': 41, 'bias': 0.16, 'home_bias': 0.22}

param_range = {
    'metaparam0': [0.015 * 0.95 ** (i - 5) for i in range(10)],
    'metaparam1': [0.025 * 0.95 ** (i - 5) for i in range(10)],
    'metaparam2': [41. * 0.95 ** (i-5) for i in range(10)],
    'bias': [0.16 + (i-5)*0.01 for i in range(10)],
    'home_bias': [0.22 + (i-5)*0.002 for i in range(10)]}

print param_range

model = Elosplit.Elosplit(data.train, data.test, param)

optimizer = MetaOpti.MetaOpti(data, param_range, param, model, lambda: model.get_cost('test'))

for i in range(100):
    optimizer.data.get(0.5)
    optimizer.mongo_opti()
    print optimizer.param

data.get(0.5)

sys.exit(0)
# Create the model
model = Elostd.Elostd(data.train, data.test, dict_param)

# Train the model
this_time = time.time()
for k in range(90):
    model.train()

# Evaluate its performance on the test set
print model.get_cost('test'), model.get_cost('train')

print int(100. * (time.time() - this_time))

# Train the model on the full dataset
data.get(1)
model.set_train_data(data.train)
for k in range(90):
    model.train()

# Get some informations on elo ratings
data.set_elos(model.get_elos())
print data.get_elos(countries=['sanmarino', 'france', 'germany', 'spain', 'romania'], times='last')
elos = data.get_elos(times='last')
elo = []
for key in elos:
    elo.append((key, elos[key][0]))
print sorted(elo, key=lambda (x, y): -y)[-5:]



# Get prediction for France vs Romania
model.set_test_data(data.create_matches([('france', 'romania', 'last')]))
print model.get_res('test')

# Close the session
model.close()
