from polyreg1 import *

max_p = 18 #18 parameters
p_range = np.arange(1, max_p + 1, 1) #creates array 1->max_p with an increment of 1
X = np.ones((n, 1)) #constructs a matrix of ones nx1
betahat , trainloss = {}, {}

for p in p_range: # p num of parameters
    if p > 1:
        X = np.hstack((X, u**(p-1))) #adds column to matrix, horizontal stack
        # this constructs the design matrix
        
    betahat[p] = solve(X.T @ X, X.T @ y)
    #via the definition of the projection matrix and the psuedo-inverse, 
    # the above solves B = (X.T X)^-1 X.T y (least squares estimator)
    # note here that using solve in numerically stable. For example, if we
    # choose to evaluate the inverse via np.linalg.inv(X.T @ X)@ X.T @ y, we face
    # an issue of floating point error as inverting the matrix directly requires
    # more floating point operations. On the other hand, solve(A, b) uses Cholesky,
    # LU and QR decomposition under the hood
    trainloss[p] = (norm(y-X @ betahat[p]) ** 2/n)
    # this computes the MSE, norm computes the Euclidean Norm

p = [2, 4, 16] #select 3 curves for comparison

#replot the points and true line and store in the list "plots"

plots = [plt.plot(u, y, 'k.', markersize = 8)[0],
         plt.plot(xx, yy, 'k--', linewidth = 3)[0]]
# note that plt.plot returns a list of Line2D objects, [0] takes the actual artist
# add 3 curves
for i in p:
    yy = np.polyval(np.flip(betahat[i]), xx)
    plots.append(plt.plot(xx, yy)[0])
# add on the fitted graphs

plt.xlabel(r'$u$')
plt.ylabel(r'$h^{\mathcal{H}_p}_{\tau}(u)')
plt.legend(plots, ('data points', 'true', '$p=2$, underfit', '$p = 4$, correct',
                   '$p = 16$, overfit'))
#plt.savefig('polyfitpy.pdf',format='pdf')
plt.show()