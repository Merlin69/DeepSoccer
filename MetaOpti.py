import random

class MetaOpti:
    def __init__(self, data, paramrange, param, model, target, M, L):
        self.data = data
        self.paramrange = paramrange
        self.param = param
        self.model = model
        self.target = target
        self.M = M
        self.L = L

    def mongo_opti(self):
        key = random.choice(self.param.keys())
        cost = 200000
        elparam = None
        m, p, t = self.paramrange[key]
        if t == 'exp':
            rangee = [m * p**(i-self.M/2) for i in range(self.M)]
        elif t == 'lin':
            rangee = [m + p * (i - self.M / 2) for i in range(self.M)]
        else:
            print 'WTF ???? (line 22 METAOPTI LOL)'
        for x in rangee:
            self.param[key] = x
            self.model.reset()
            self.model.set_params(self.param)
            for i in range(100):
                self.model.train_res()
            print self.model.get_cost('train'), self.model.get_cost('test')
            temp = self.target()
            print key, x, 'current_los', temp
            if temp < cost:
                cost = temp
                elparam = x
        self.param[key] = elparam
        if t == 'exp':
            self.paramrange[key] = elparam, p**self.L, t
        else:
            self.paramrange[key] = elparam, p * self.L, t
        print 'final', cost




