import numpy as np
from numpy.random import rand , randn
from numpy.linalg import norm , solve


import matplotlib.pyplot as plt

#function to generate the data
def generate_data(beta, sig, n):
    u = np.random.rand(n, 1) 
    #generates a nx1 vector of elements uniformly on (0,1) (iid.)
    y = (u ** np.arange(0,4)) @ beta + sig * np.random.randn(n,1)
    #we now transform this into a polynomial 
    # np.arange(0, 4) = [0, 1, 2, 3], we are creating a power basis of degrees 
    # 0-3
    # u ** np.arange(0, 4), NumPy broadcasting is used here, resulting in X \in 
    # R^(n x 4), where each column is u^d, d = 0-3: X_ij = u_i^j
    # We call X = [1, u, u^2, u^3] (Vandermonde Matrix) for polynomial regression
    # @ beta: performs matrix multiplication. in this case, beta \in R^(4 x 1)
    # so, here y_noiseless = XB \in R^(n x 1)
    # sig * np.random.randn(n, 1): adds i.i.d Gaussian noise ~N(0, \sigma^2 I_n)
    # note here that random.randn(n, 1) generates samples from the standard normal
    # distribution, so sig tells us the standard deviation
    return u, y

np.random.seed(12) #fixing the seed here fixes the internal state of the random 
# number generator so we always get the same random numbers, reproducibility

beta = np.array([[10, -140, 400, -250]]).T #transpose, establishes the coefficients
n = 100 #this tells us the number of samples
sig = 5 # tells us the standard deviation
u, y = generate_data(beta, sig, n)
xx = np.arange(np.min(u), np.max(u) + 5e-3, 5e-3)
#this creates a dense, evenly spaced grid of input values between min and max of u
# with spacing of 0.005.
yy = np.polyval(np.flip(beta), xx)
#here, we are evaluating the polynomial defined by the coefficients beta at the 
# input points xx, however, np.polyval assumes coefficients are ordered from highest to 
# lowest degree, so we have to np.flip the vector beta

plt.plot(u, y, '.', markersize = 8) #data
plt.plot(xx, yy, '--', linewidth = 3) #predictor model

plt.xlabel(r'$u$')
plt.ylabel(r'$h^*(u)$')
plt.legend(['data points', 'true']) #first plotted object is data points, second
#plotted is true
plt.show()