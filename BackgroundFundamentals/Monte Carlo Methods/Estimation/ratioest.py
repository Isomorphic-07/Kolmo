# Bootstrap method, ratio estimator of Markov Chain
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from numba import jit  # speeds up numerical operations with Just-In-Time compilation

np.random.seed(123)

n = 1000
P = np.array([[0, 0.2, 0.5, 0.3],
              [0.5 ,0, 0.5, 0],
              [0.3, 0.7, 0, 0],
              [0.1, 0, 0, 0.9]])  # Transition probability matrix

r = np.array([4.0, 3.0, 10.0, 1.0])  # reward vector
rho = 0.9  # discounting factor — presence of a discounting factor encourages the agent to 
# prefer short term rewards; think of it like time preference in economics

@jit(nopython=True)  # compiles the function to fast machine code (no Python overhead)
def generate_cyclereward(n, P, r, rho):
    Corg = np.zeros(n)  # stores cycle times (lengths until regeneration)
    Rorg = np.zeros(n)  # stores total discounted rewards per cycle

    for i in range(n):
        t = 1
        xreg = 1  # regenerative state (state 1), process restarts here
        reward = r[0]

        # Sample the next state from current xreg using inverse CDF sampling
        probs = np.cumsum(P[xreg - 1])  # builds cumulative distribution: e.g., [0.2, 0.7, 1.0]
        x = np.searchsorted(probs, np.random.rand()) + 1  # returns first index where rand < CDF
        #alternatively, we can do  x= np.amin(np.argwhere(np.cumsum(P[xreg-1,:]) > np.random.
        #rand())) + 1

        # Run cycle until return to regenerative state
        while x != xreg:
            t += 1
            reward += rho**(t - 1) * r[x - 1]  # accumulate discounted reward
            probs = np.cumsum(P[x - 1])  # transition from current x
            x = np.searchsorted(probs, np.random.rand()) + 1  # sample next state

        Corg[i] = t
        Rorg[i] = reward

    return Corg, Rorg

# Simulate original sample
Corg, Rorg = generate_cyclereward(n, P, r, rho) #ensures JIT safe
Aorg = np.mean(Rorg) / np.mean(Corg)  # original long-run average reward

# Bootstrap estimation
K = 5000  # number of bootstrap replications
A = np.zeros(K)  # stores each bootstrap estimate

for i in range(K):
    ind = np.random.choice(n, size=n, replace=True)  # bootstrap sample (with replacement)
    C = Corg[ind]
    R = Rorg[ind]
    A[i] = np.mean(R) / np.mean(C)  # bootstrap estimator of long-run average reward

# Plotting density of bootstrap distribution
plt.xlabel('long-run average reward')
plt.ylabel('density')
sns.kdeplot(A, shade=True)
plt.show()