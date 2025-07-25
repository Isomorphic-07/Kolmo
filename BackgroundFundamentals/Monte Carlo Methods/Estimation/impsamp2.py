from impsamp1 import *

lam = 0.1
g = lambda x1, x2: lam*exp(-sqrt(x1**2 + x2**2)*lam)/sqrt(x1**2 + x2
**2)/(2*pi)
U = rand(N,1)
V = rand(N,1)
R = -log(U)/lam #quantile function of exponential dist
X1 = R*cos(2*pi*V)
X2 = R*sin(2*pi*V)
Z = H(X1, X2)*f/g(X1, X2)


