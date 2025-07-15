"""
1. Naive Recursive: O(a^n) or O(n!)
2. Top-Down DP-Memoization (Optimise 1., via a cache)
3. Bottom-up DP- Tabulation
4. Constant space of (3)

Lets solve the Fibonacci Problem
"""

#Naive Recursion
def fib(n):
    if n == 0 or n == 1:
        return n

    return fib(n-1) + fib(n-2) #in leet code, solution are implemented in a class, so you would write self.fib(n-1) + self.fib(n-2)

"""
notice in the above, we get multiple call of the same function:
                                        f(6)
                            f(5)                    f(4)
                    f(4)            f(3)    f(2)            f(3)
                                         ...
So it is very inefficient. So notice, the tree always have leaves as f(1) or f(0). Notice, we are doubling each time, so we
are performing at time complexity of O(2^n). We optimize this via Top Down Memoization (really easy):
idea is that when we have already done a function call, don't do the operation again as we have already performed and so
from the cache where we store the value of the function call, we can perform the function in O(1)

We can do this via a dictionary, where we store the values to each input of the function:
memo (cache) = {0: f(o), 1: f(1), ...}. Top down cause we start at the top. So this allows us to achieve O(n) in time complexity
as we always refer back to the value stored in the cache. Space complexity is O(n) due to the height of the recursive call stack
"""

#memoization
def fibTop(n):
    memo = {0:0, 1:1}

    def f(x):
        if x in memo:
            return memo[x]
        else:
            memo[x] = f(x-1) + f(x-2)
            return memo[x]

    return f(n)

"""
Bottom Up elliminates recursion. (This is the standard way to solve quant probability questions via DP). You start at the base
case where the problem is easier to solve and move up to the problem that we want to solve. Which is why we call it tabulation
as we keep storing the value for future use. So we create a large array in the beginning stored with 0's and initialise the
base case in the array. Here, time complexity and space is still O(n). Preferred as we avoid recursion
"""

def fibBot(n):

    if n == 0:
        return 0
    if n == 1:
        return 1

    dp = [0] * (n+1)
    dp[0] = 0
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-2] + dp[i - 1]

    return dp[n]

"""
We can optimise further as we still keep past values in memory, so we can pop off elements in the array that we don't need
anymore. Time: O(n), Space: O(1)
"""

def fibBotOpt(n):

    if n == 0:
        return 0
    if n == 1:
        return 1

    prev = 0
    cur = 1

    for i in range(2, n + 1):
        """
        prevCopy = cur
        cur = cur + prev
        prev = prevCopy
        """
        prev, cur = cur, prev+ cur #very clever

    return cur

#A final note: MATHEMATICAL CALCULATIONS ARE ALWAYS FASTER, SO IF A MATHEMATICAL EQUATION CAN BE DERIVED, IT IS MORE EFFICIENT:
def fibMath(n):
    golden_ratio = (1 + (5 ** 0.5)) / 2
    return int(round((golden_ratio ** n) / (5 ** 0.5)))