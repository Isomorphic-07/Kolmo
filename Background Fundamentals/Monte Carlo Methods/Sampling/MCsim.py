"""
Simulate a Markov Chain
- input: number of steps N, initial pdf f_0 and transition density q
- first draw X_0 from initial pdf f_0
- for t from 1 to N: Draw X_t from distribution corresponding to the density q
- return all X_i
"""
import numpy as np
import matplotlib.pyplot as plt

n = 101
P = np.array([[0, 0.2, 0.5, 0.3], [0.5, 0, 0.5, 0], [0.3, 0.7, 0, 0],
              [0.1, 0, 0, 0.9]])
#construct transition matrix

x = np.array(np.ones(n, dtype=int))

#x[0] = 0 #this initialises our initial position start at zero definitely

# Initial distribution: 50% in state 1, 50% in state 2
f0 = np.array([0, 0.5, 0.5, 0])
x[0] = np.min(np.where(np.cumsum(f0) > np.random.rand()))
# this selects where we will begin

for t in range(0, n-1):
    x[t + 1] = np.min(np.where(np.cumsum(P[x[t],:]) > np.random.rand()))
    #this performs quantile function

x = x + 1 #add 1 to all elements of vector x.
plt.plot(np.array(range(0,n)), x, 'o')
plt.plot(np.array(range(0,n)), x, '--')
plt.show()