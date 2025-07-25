#Traditional: Binary used to search (lookup) for a value in an array O(n)
#2 pointer technique: first look at the median position (L + 2)//2
#If the median position is not what we are looking for, check what this number is
#and observe whether the target value if > or < this number. (note that this is
# only for arrays in ascending order). this allows us to cancel out portions of the array
# then do this again by changing the left (L) and right (R) pointers. Note we use
# the term pointers losely here, we really mean indices.

#however, note the median formula, when used for larger numbers can result in 
# integer overflow, so we actually want to avoid. So we use: M = L + (R- L)//2




#Condition Based: occurs when say for example, we have an array [T T T F F F]
# say we want to find the first time it is F. Again, do M = (L + R)//2. If M is F, 
# then move R to where M, else if M is T, move L where M is. Keep doing this until 
# L=M, at which we are at the first position

#Time: notice that this method is dependent on 2^n, and so we are looking at O(log n)
# i.e with 1024 elements, it would take 10 tries to keep halving down

#Space: O(1) as you really only store 3 elements: L,M,R

A = [-3, -1, 0, 1, 4, 7]
print('hi')
#naive
if 8 in A:
    print(True)

#traditional: time O(log n), note, we stop checking when the L and R's cross
def check(array, target):
    L = 0
    R = len(array) - 1
    while L <= R:
        M = L + (R - L)//2
        if array[M] == target:
            return(True)
        elif array[M] > target:
            R = M - 1
        elif array[M] < target:
            L = M + 1
        
    return(False)

print(check(A, 8))

#condition
B = [False, False, False, False, True, True, True]

def bin_search_condition(arr):
    L = 0
    R = len(arr) - 1
    
    while L < R:
        M = L + (R - L)//2
        if arr[M]:
            R = M
        else:
            L = M + 1
            
    if arr[L]:
        return L
    else:
        return False
print(bin_search_condition(B))
