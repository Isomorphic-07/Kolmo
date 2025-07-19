from mcint import *

#this example performs the same integrtion as mcint via variance reduction
#remember, we want to find the expected value of y
Yc = np.sum(x**2, axis =1) #control variable data, establishes x_1^2 + x_2^2 + x_3^2
yc = 3 #true expectation of control variable as x_1,x_2,x_3 ~ N(0,1)^3
C = np.cov(y, Yc) #sample covariance matrix
cor = C[0][1]/np.sqrt(C[0][0]*C[1][1]) #correlation between y and Yc
alpha = C[0][1]/C[1][1] 

est = np.mean(y-alpha*(Yc-yc))
RECV = np.sqrt((1-cor**2) * C[0][0]/N)/est #relative error, derived from minimised variance
# refer to notes for proof

print('Estimate = {:3.3f}, CI = ({:3.3f}, {:3.3f}), Corr = {:3.3f}'.format(est,
        est*(1-z*RECV), est*(1+z*RECV), cor))