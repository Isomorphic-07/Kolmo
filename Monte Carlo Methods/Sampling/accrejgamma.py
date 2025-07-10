from math import exp, gamma, log
from numpy.random import rand

alpha = 1.3
lam = 5.6
f = lambda x: lam**alpha * x**(alpha-1) * exp(-lam*x)/gamma(alpha)
#establish the pdf of the gamma function: f(x)

g = lambda x: 4*exp(-4*x)
#establishes the upper bound function for acceptance rejection: Cg(x)
C = 1.2

found = False
while not found:
    #generate X from g, we can do this via inverse transform
    #x = 4e^{-4y}
    #y = -log(x/4)*1/4
    x = - log(rand()) / 4
    if C*g(x)*rand() <= f(x):
        found = True
        
print(x)