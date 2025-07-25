#static arrays are a contiguous block of memory, established in the start when 
#array is initialised (mutable)

#dynamic array (list), can change size (via a static array, by copying and re-establishing the array)
#when appending to dynamic array, can be O(1), however, if it involves expaning the dimensions
#it is O(n) as we have to create a copy of the original. What python does is when
#it expands/appends (end) to an array, it doubles it size (2^n), so most of the time, it is 
# O(1) (amortized), doubling (x2) is chosen to reduce wasted memory especially early on
# , reduces cache inefficiencies, and a larger multiplier would result in deminishing returns
# manipulating (add/delete) anything not at the end is O(n)


#strings are immutable, and so most operations are O(n), except for random access

A = [1, 2, 3]
A.append(5)
A.pop() #deletes at end of array
A.insert(2, 5) #position, value
