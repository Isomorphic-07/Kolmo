"""
-Make decisions
- Perform Recursion
- Base case
- undo decisions

Often to perform an exhaustive search (when you see all solutions)
Given an integer array nums of unique elements, return all possible subsets 
(power set).

i.e [1, 2, 3]
Let us first establish the descisions that can be made here, namely, we can either
choose to accept a value from the array into the subset array or reject it. This forms
a decision tree where at the first level, we either choose or reject 1 into the subset
array. In the next level we choose to either accept or reject 2 to [], [1]. Then next level,
accept or reject 3 to [], [2], [1], [1, 2]. So we have established how this would
be done, but for the computer, it would use a DFS as we are only concerned with the
lower level arrays:

The recursive backtrack occurs when the DFS algorithm hits one of the solutions
We first initialise a results RES (stores all solutions) and a sol = [] (a template for recursive back tracking)
used to append into and pop out (undoing changes).

Via DFS, once we hit a leaf node, give a copy of the solution into RES. At a node, we can choose
to pick a value, i.e pick 3, and this would go into our sol array: sol = [3], recurse 
down the path, if we hit a base case, append this into RES = [[], [3]].

We then pop the 3 out from sol, going back to the parent node. So sol just stores in
what the solution can be to be appended into results
"""

def subsets(nums):
    n = len(nums)
    res, sol = [], []
    
    def backtrack(i):
        if i == n:
            #when we are the end and have a solution
            res.append(sol[:]) #we store a sol copy, we don't want to but the actual instance into the res
            return
        
        # Don't pick nums[i]
        backtrack(i + 1)
        
        #Pick nums[i]
        sol.append(nums[i]) #choose nums[i]
        backtrack(i + 1) #move forward
        #now, we want to undo our choice (undo the changes, recursively backtrack)
        sol.pop()
    
    backtrack(0) #start at very beginning
    return res

#Time: notice in every level, we are doubling, resulting in O(2^n)
#Space: recursion depth here would take extra space, we have to store these call stacks,
#here, the depth of the tree is n, and so O(n)
    