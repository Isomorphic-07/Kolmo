"""
The idea of insertion sorting is that you have a sorted area of the array (usually the 
first position) and overtime, we insert elements from the unsorted region into this sorted region
So we begin i at 1, and j at this i. We compare, is a[j-1] < a[j], swap accordingly. If
this condition is fine, then we move i and j forward. Else if a swap is involved, we move j back to j = 1
and check if we must swap any elements to ensure a sorted region of the array
Time: O(n^2)
Space: O(1)
"""

B = [-5, 3, 2, 1, -3, -3, 7, 2, 2]

def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        for j in range(i, 0, -1): #go from i down to excluding 0
            if arr[j-1] > arr[j]:
                arr[j-1], arr[j] = arr[j], arr[j-1]
            else:
                #correct format
                break

insertion_sort(B)
print(B)