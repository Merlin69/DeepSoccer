import tensorflow as tf


class Cost:
    def __init__(self, model, name, costs={}):
        self.costs = costs
        self.model = model
        self.name = name

    def define_logloss(self, slices, prediction, feature, feature_size):
        rindices = [i + 1 for i in range(feature_size)]
        for s in slices:
            probabilities = prediction[s] * self.model.data.tf_slices[s][feature]
            probabilities = tf.reduce_sum(probabilities, reduction_indices=rindices)
            self.costs[s] = tf.reduce_mean(-tf.log(probabilities + 1e-9))

    def apply(self, other, fun):
        if self.model != other.model:
            print('different model for cost addition')
        result = Cost(self.model, '')
        if type(other) in [type(0), type(0.)]:
            for s in self.costs.keys():
                result.costs[s] = fun(self.costs[s], other)
                result.name = self.name + '+' + str(other)
        else:
            slices = list(set(self.costs.keys()) & set(other.costs.keys()))
            result.name = self.name + '+' + other.name
            for s in slices:
                result.costs[s] = fun(self.costs[s], other.costs[s])
        return result

    def __add__(self, other):
        return self.apply(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        self.apply(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.__mul__(other)

    def get(self, s):
        return self.costs[s]



