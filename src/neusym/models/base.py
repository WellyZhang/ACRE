# -*- coding: utf-8 -*-


class BaseModel(object):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    
    def train(self, context):
        raise NotImplementedError

    def test(self, query):
        raise NotImplementedError

    def predict(self, prob):
        if prob < self.lower:
            return 0
        elif prob < self.upper:
            return 1
        else:
            return 2