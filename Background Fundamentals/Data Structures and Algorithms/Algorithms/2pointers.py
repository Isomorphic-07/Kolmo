"""
Squeeze algorithm: we have a L index beginning and R index at the end and we begin
to squeeze in towards the centre

Motivation: consider the squares of a sorted array problem where where have a
non-decreasing array, but we want to return an array of the squares of each number sorted in 
non-decreasing order. We can do this by ofc first squaring all elements in place, then perform
merge/quick sort to convert to ascending order: O(n log n). WE CAN BEAT THIS !

We first notice, that the largest values will be on the left and right of the square
array: i.e:
[-4, 1, 0, 3, 10] -> ^2 -> [16, 1, 0, 9, 100]
So say we're building up a new array, comparing between L and R, which is bigger will
be the first element in array. So in this case 100, and we decrememnt index of R.
Compare between L and R, notice 16 > 9, so we append 16 to array and increment index of L.
Continue until we hit the middle when R = L and we append the last element (L <= R)

"""

def sortedSquares(self, nums):
    left = 0
    right = len(nums) - 1
    result = []
    
    while left <= right:
        if abs(nums[left]) > abs(nums[right]):
            result.append(nums[left] ** 2)
            left += 1
        else:
            result.append(nums[right] ** 2)
            right -= 1
            
    result.reverse()
    return result

#Time: O(n)
#Space: O(1) (assuming that the created array of result is not counted as extra space, else O(n))
# we are just using the output (reequired) space, nothing extra