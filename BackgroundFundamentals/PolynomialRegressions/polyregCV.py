from polyreg3 import *

K_vals = [5, 10, 100] #number of folds
cv = np.zeros((len(K_vals), max_p)) #cv loss, creates a matrix of 0's with 3 rows
# (number of K_vals) and max_p paramters columns
X = np.ones((n,1))

for p in p_range:
    if p > 1:
        X = np.hstack((X, u**(p-1)))
    j = 0
    for K in K_vals:
        loss = [] #store our loss values
        for k in range(1, K+1):
            # integer indices of test samples
            test_ind = ((n/K)*(k-1) + np.arange(1, n/K + 1) -1).astype('int')
            # test indices: the above provides an array of indices of data that will be in the
            # kth fold 
            train_ind = np.setdiff1d(np.arange(n), test_ind)
            #train indices: selects all indices not in the test indices, essentially
            # setdiff1d finds indices in the array np.arange[n]: [0, n-1] that
            # are not in the test_ind
            
            X_train, y_train = X[train_ind, :], y[train_ind, :]
            #X[train_ind, :] selects rows of the input feature matrix X corresponding
            # to the train_ind indices and : selects all columns 
            X_test, y_test = X[test_ind, :], y[test_ind, :]
            
            #fit the model and evaluate test loss
            betahat = solve(X_train.T @ X_train, X_train.T @ y_train)
            loss.append(norm(y_test - X_test @ betahat) ** 2)
            
        cv[j, p-1] = sum(loss) / n
        #computes the cross validation 
        j += 1

plt.plot(p_range, cv[0, :], 'k-.')
plt.plot(p_range, cv[1, :], 'r')
plt.plot(p_range, cv[2, :], 'b--')
plt.show()