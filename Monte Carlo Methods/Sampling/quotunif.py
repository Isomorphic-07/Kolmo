#Resampling: an iid sample \tau from some unknown unknown cdf F reporesents our
# best knowledge of F is we make no further a priori assumptions. If it is not possible to simulate
# more samples from F, the best way to repeat the experiment is to resample from 
# the original data by drawing from the empirical cdf F_n
"""
Input: original iid sample x_1,...,x_n and sample size N
output: iid sample X*_1, ..., X*_n from empirical cdf
for t = 1 to N:
- Draw U ~ U[0,1]
- Set I = upperfloor(nU)
- Set X*_t = x_I
Return X*_1, ..., X*_N

in this example, we look a quotient of uniforms: Let U_1,..., U_n, V_1, ...,V_n
be iid U(0,1) r.v and define X_i = U_i / V_i, i = 1, .., n. Suppose, we want to
investigate the distribution of sample median \tilde{X} and sample mean of the random data T:
[X_1, ..., X_n]
"""
import numpy as np
from numpy.random import rand, choice
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

n = 100
N = 1000
x = rand(n)/ rand(n) #data, creates an array
med = np.zeros(N)
ave = np.zeros(N)
for i in range(0, N):
    s = choice(x, n, replace = True) #resampled data
    med[i] = np.median(s)
    ave[i] = np.mean(s)
    
med_cdf = ECDF(med) #ECDF is an object, constructs empirical cumulative distributive
                    #function, calculated via f_n(t) = 1/n \sum_i=1^n {I(x_i < t)}
ave_cdf = ECDF(ave)

plt.plot(med_cdf.x, med_cdf.y)
plt.plot(ave_cdf.x, ave_cdf.y)
plt.show()
