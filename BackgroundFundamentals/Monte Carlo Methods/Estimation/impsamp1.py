import numpy as np
from numpy import exp, sqrt, sin, pi, log, cos
from numpy.random import rand
b = 1000
H = lambda x1, x2: (2*b)**2 * exp(-sqrt(x1**2+x2**2)/4)*(sin(2*sqrt(
x1**2+x2**2))+1)*(x1**2 + x2**2 < b**2) #restricts the domain to a disk of radius b
f = 1/((2*b)**2)
N = 10**6
X1 =-b + 2*b*rand(N,1) #generates N samples 
X2 =-b + 2*b*rand(N,1)
Z = H(X1,X2)
estCMC = np.mean(Z).item() # to obtain scalar
RECMC = np.std(Z)/estCMC/sqrt(N).item()
print('CI = ({:3.3f},{:3.3f}), RE = {: 3.3f}'.format(estCMC*(1-1.96*
RECMC), estCMC*(1+1.96*RECMC),RECMC)) #95% CI