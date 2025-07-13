"""
Bubble sort is a sorting algorithm that aims to reallocate an array in ascending
order by starting i at i = 1, checking whether a_0 < a_1, swap terms accordingly then
increment i. This would be done again in another cycle until in the final cycle,
there are no swaps needed leading to the array being in ascending order. Results in
time: O(n^2) and space of O(1) (in place sorting as we are not creating a new array
and we only compare 2 values at a time)
"""

A = [-5, 3, 2, 1, -3, -3, 7, 2, 2]

def bub_sort(arr):
    flag = True #tells us we are not done yet
    n = len(arr)
    while flag:
        flag = False #assume we are done, but once there is a swap, we change it back
        for i in range(1, n):
            if arr[i] < arr[i-1]:
                flag = True
                """
                copy = arr[i]
                arr[i] = arr[i-1]
                arr[i-1] = copy
                """
                arr[i -1], arr[i] = arr[i], arr[i-1]
                
                
bub_sort(A)
print(A)