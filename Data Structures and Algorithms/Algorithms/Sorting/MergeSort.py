"""
The divide and conquer algo. 

DIVIDE: Let M be the median index of the array
M = (len(arr)-1)//2. From this index, split the array into 2 segments. If odd number
of elements, left and right hand side of M (right hand side array will include M)
Keep splitting these arrays by half on going until we split it into individual elements
which builds our base case.

CONQUER: Now look at a pair of single elements and form a sorted array of dimension 2 from
this. This creates a bunch of sorted arrays that we now need to merge. To perform the merge,
consider the example:
[-5, 3] <-> [1, 2]
 L           R
Look at L and R, which do we prefer?, L, so we increment L and add -5 into merged array:
[-5 x x x]
[-5, 3] <-> [1, 2]
     L       R
Do this step again: (a crucial idea is when R is at 2, when we accept the R, it will
increment to an index outside the array, signalling that we just take what ever is left
from the other array and append it to the end of our merged array)
[-5, 1, 2, 3]. This idea is applied until we have a fully sorted
array.

Thinking about this, this actually forms a tree!, where the height of the tree is 
log_2 {n}. Additionally, to traverse each of the elements in each level (there are n elements
in each level), this is O(n) (to merge). So the total time complexity if O(n log n)

Note that our recursive call stack takes up space and the height is log_2{n}, the space
complexity is O(log n). However, in our example it will be O(n) as we store all the elements in
each progressed level
"""

def merge_sort(arr):
    n = len(arr)
    
    if n == 1:
        return arr
    
    #DIVIDE!!!
    m = (n-1)//2
    L = arr[:m] #Storing O(n)
    R = arr[m:]
    
    #lots of recursion
    L = merge_sort(L)
    R = merge_sort(R)
    l, r = 0, 0 #here, we will slide these indices
    L_len = len(L)
    R_len = len(R)
    
    sorted_arr = [0]*n
    i = 0 #used to denote the index in sorted_arr
    
    #CONQUER (MERGING)
    while l < L_len and r < R_len:
        if L[l] < R[r]:
            sorted_arr[i] = L[l]
            l += 1
        else:
            sorted_arr[i] = R[r]
            r += 1
        
        i+=1
        
    #checks if anything is left in L
    while l < L_len:
        sorted_arr[i] = L[l]
        l += 1
        i += 1
        
    #checks if anything is left in L
    while r < R_len:
        sorted_arr[i] = L[r]
        r += 1
        i += 1
        
    return sorted_arr

