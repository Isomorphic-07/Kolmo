"""
Variable Length: (window can change in length) given an array/string, in the context
of subarray/substring.

We look at a problem concerning Longest Substring without repeating characters:
Given a string s, find the length of the longest substring (a contiguous non-empty sequence
of characters within a string) without repeating characters

Windowing is created via and L and R pointer, which both initially start at 0, R
and L are moved up accordingly to change the size (contract/expand) the window, and
thus, overall it is an O(n) operation as we just slide L and R across.
We want to have a set as well to store unique elements.

First, we need to establish what makes a valid window. In this case, a valid window
occurs when there are no duplicate characters. With each window, we record in the element/s
that are in the window in the set to keep track of the longest length. 

So while we have a valid window, we would keep incrementing R, if window still valid,
add in the element into the set and update the length. Once we get an invalid window,
we must remove the repeated element in the set, then we then increment L and check (usually
add the element at R back in the set or remove more).  
We only update the length if 
we get a valid window of length greater than the previously recorded length. Note
window length is:
w = (R- L) + 1
We complete once R goes to index outside the string/array

"""

def lengthOfLongestSubstring(self, s):
    l = 0 #index L
    longest = 0
    sett = set() #set to store unique characters in window
    n = len(s)
    
    #O(n)
    for r in range(n):
        #r is in for loop for implementation of linear time
        while s[r] in sett:
            #when window is invalid, this loop here is O(n) as well
            sett.remove(s[l])
            l += 1
        
        w = (r - l) + 1
        longest = max(longest, w)
        sett.add(s[r])
        
    return longest
#Time: O(n)
#Space: O(n) (since we use a set for this problem, otherwise O(1))

"""
Fixed-length sliding window: Maximum Average Subarray, we are given an integer
array nums consisting of n elements and an integer k. Find a contiguous subarray whose length is equal to k
that has the maximum average value and return this value. (INTERESTING!!!)

So we can see here that k establishes the fixed length of this window. Here, we want
to hold max_avg (initialise to -\infty) and curr_sum = 0. We begin an index at 0,
recording in the curr_sum of the sum of each of the elements, keep incrementing 
the index until we get the length of k as our window. Once we get here, we calculating the
current avg = curr_sum/k

Now lets slide the window, we add whats on the right by incrementing the index and
lose the first element, acheiving a new window, keep doing this and update max_avg 
accordingly
"""

def findMaxAverage(self, nums, k):
    n = len(nums)
    cur_sum = 0
    
    for i in range(k):
        cur_sum += nums[i]
        
    max_avg = cur_sum / k
    
    for i in range(k, n):
        cur_sum += nums[i]
        cur_sum -= nums[i-k]
        max_avg = max(cur_sum / k, max_avg)
        
    return max_avg

#Time: O(n)
#Space: O(1), we aren't really storing anything