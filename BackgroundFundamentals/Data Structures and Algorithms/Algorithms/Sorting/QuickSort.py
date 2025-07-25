"""
Pick a pivot, any index. Cut up the array into 3 sections, what is less or equal
to the value of the pivot, the pivot and what is more than the value of pivot. You then
perform this operation again on the 2 other sections (pivot section is the base case).
It is quite important about what pivot is chosen, we can get bad pivots where for
example everything in the array is greater than/ less than the pivot resulting in 
unecessary operations. Empty lists are also base cases

We do this until we get 1 or 0 element arrays. Then we just merge these components
together. However notice via our construction, we can simply concatenate these
components together and it becomes sorted

Time Complexity: At each level of deconstruction, we merge n elements. In the best
case scenario there are log_2{n} levels, so O(n log n). However, we can get bad pivots 
where we would only extract one element out a time from the deconstructed array,
resulting in O(N^2). In the average case, O(n log n)*

Space Complexity: space of recursive call stack, O(log n) (each recursion stage is
stored in the stack)
"""
#Space: O(N)

E = [-5, 3, 2, 1, -3, -3, 7, 2, 2]

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    p = arr[-1] #we choose pivot to be the last index
    
    #list comprehension
    L = [x for x in arr[:-1] if x <= p]
    R = [x for x in arr[:-1] if x > p]
    
    L = quick_sort(L)
    R = quick_sort(R)
    
    return L + [p] + R

print(quick_sort(E))


