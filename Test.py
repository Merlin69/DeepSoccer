import csv
import sys
import pylab as pl
import numpy as np
import math


def get_money(team):
    money1 = 0
    money2 = 0
    nb_match = 0
    first_line = False
    with open('Datas/DataWithOddsD1.txt', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if row and first_line:
                [date, h, a, r] = row[1:5]
                if 20 > int(date[-2:]) > 10:
                    [c1h, c1d, c1a, c2h, c2d, c2a] = map(float, row[-7:-1])
                    h = int(h)
                    a = int(a)
                    if h == team or a == team:
                        money1 -= 2
                        money2 -= 2
                        nb_match += 1
                        if r != 'D' and r!= 'A' and r != 'H':
                            print 'WARNING'
                        if r == 'D':
                            money1 += c1d
                            money2 += c2d
                    if h == team and r == 'A':
                        money1 += c1a
                        money2 += c2a
                    if a == team and r == 'H':
                        money1 += c1h
                        money2 += c2h
            elif not first_line:
                first_line = True
    return money1, money2, nb_match


def get_percent(nb, money):
    if nb != 0:
        return 100. * (money / (2. * nb))
    else:
        return 0.

for i in range(42):
    print get_money(i+1)

z1 = np.array([get_money(i+1)[0] for i in range(42)])
z2 = np.array([get_money(i+1)[1] for i in range(42)])

p1 = np.array([get_percent(get_money(i+1)[2], get_money(i+1)[0]) for i in range(42)])
p2 = np.array([get_percent(get_money(i+1)[2], get_money(i+1)[1]) for i in range(42)])

zero = np.array([0.] * 42)
X = np.array(range(42))
pl.plot(X, p1)
pl.plot(X, p2)
pl.plot(X, zero)
pl.show()
