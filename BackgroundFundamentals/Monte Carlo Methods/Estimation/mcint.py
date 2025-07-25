import numpy as np
from numpy import pi

#performing integration
c = (2*pi)**(3/2)
H = lambda x : c * np.sqrt(np.abs(np.sum(x, axis=1))) #defines what we are trying to take the
#expected value of


N = 10 ** 6
z = 1.96 #z-score at a 95% confidence interval
x = np.random.randn(N, 3)
y = H(x)

mY = np.mean(y)
sY = np.std(y)
RE = sY/mY/np.sqrt(N) #relative error
print('Estimate = {:3.3f}, CI = ({:3.3f}, {:3.3f})'.format(mY, mY*(1-z*RE),mY*(1+z*RE)))