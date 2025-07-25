"""
Place index i at first position, j = i. We want the minimum value at the beginning
of the array, so we increment j looking for the smallest value. Once smallest value is found 
(M), swap with a_0. Now, we have the smallest value in the first position, now increment i = j = 2
and do it again, send j up to look for the smallest value in the array and replace accordingly

Time: O(n^2)
Space: O(1)
"""

C = [-3, 3, 2, 1, -5, -3, 7, 2, 2]

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        minIndex = i
        for j in range(i + 1, n):
            if arr[j] < arr[minIndex]:
                minIndex = j
                
        arr[i], arr[minIndex] = arr[minIndex], arr[i]

selection_sort(C)
print(C)