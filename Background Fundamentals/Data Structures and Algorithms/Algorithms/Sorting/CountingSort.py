"""
For look for max value across array (Maxx). We create a new array with Maxx + 1
positions with 0's as elements. We call this array counts as it stores in each position
the amount of occurence for each number with the respective index. i.e index 4 tells
the value of how many times 4 occurs in the array (notice here that this is applicable so 
far for only positive numbers, but we can extend this array out (by finding the minimum as
well) for negatives).

Once we complete this counts array, we essentially go through it and start replacing
values in the original array such that we sort it

Time complexity: notice, we create an array of K = Maxx many values, and to
find the maximum value, we must traverse N many elements, then figuring out the values
of the counts array, an N operation. Then we go through the array and swap with values
in the counts array which is also N operation. So O(K + N), which tells us that 
if the maximum value is really large, this algorithm is not efficient. This means
that we can get approximatly a O(N) is the max value is really small

Space Complexity: Since we overwrite the initial array, so the only space we are 
using is the counts array, so O(K)
"""
F = [5, 3, 2, 1, 3, 3, 7, 2, 2]
def counting_sort(arr):
    maxx = max(arr)
    n = len(arr)
    """
    for i in range(n):
        if arr[i] > max:
            max = arr[i]
    """
    counts = [0] * (maxx + 1)
    for x in arr:
        counts[x] += 1
        
    i = 0 #index going through the array
    for c in range(maxx + 1):
        while counts[c] > 0: #looking at the value in counts
            arr[i] = c
            i += 1
            counts[c] -= 1
    

counting_sort(F)
print(F)