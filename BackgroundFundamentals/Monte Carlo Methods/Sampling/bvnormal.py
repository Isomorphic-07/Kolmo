import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

N = 1000
r = 0.99 #correlation coefficient between the 2 bivariate variables
Sigma = np.array([[1,r], [r,1]]) #forms  2x2 covariance matrix
B = np.linalg.cholesky(Sigma)
x = B @ randn(2,N) #here the mean is 0
plt.scatter([x[0,:]],[x[1,:]], alpha = 0.4, s = 4)
plt.show()
# note if we wish to find the inverse transform, we can do so via
# np.min(np.where(np.cumsum(p) > np.random.rand()))
# here the np.random.rand() generates ~U[0,1]
# so finding the min p where the cumulative probability > U gives us inverse transform