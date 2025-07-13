"""
Heaps (priority queues) are a particular type of binary tree. In this example,
to express the tree as an array, we show that the left child is in the 2i + 1, and right
is in 2i + 2 positions.

Min Heap (done via minheapify), where the minimum is the root node, where all parents
are smaller than its children. when we perform a heap pop, this pops out the top minimum 
value, and we promote the smaller child. This operation is thus just going through one
element of each level and so is O(log n): Heap pop (extract min)

Insert (Heap push), we put the value at a leaf of the heap, and check if the position is 
valid, if it is more than parent, then it can stay, however if not, it rises up and 
we keep check the value of the parent, resulting in O(log n) (assuming roughly height balanced)

Heap peek: O(1) top root node

Heap sort: where you repeatedly perform heap pop to get an array of the elements 
in ascending order: O(n log n), in terms of space, we can just do O(1), but depends on what we want
to store

Heapify: converts a binary tree to a heap,in this case we consider the min heap
ignore the leaves and look at the rightmost bottom node. We now sift down, which is where
we look at the node we are at, look at its children, and check whether we should change this position
via swapping, then look at the next node on the left. Then once we finish a level,
we go up to the next level and look at the right most node and do it again. This
gives O(n) in time complexity

Proof: Lets provide a simpler intuitive interpretation of heapify, we use a bottom-up
method where be begin from the last non-leaf node at index \floor{n/2}. An incorrect
oversight that can occur here from the sift down is that in each sift down, it has O(log n),
and then we do this for all other nodes so maybe O(n log n), but WRONG! What we need to consider
is that each level of the heap has exponentially more nodes, byt each node at that level can
only sift down a few levels, so we weigh the number of nodes by the height they can move.
Assuming a balanced tree:

h = height of heap
n = no. of nodes
d = number of nodes at depth d

We note that in a complete tree, the number of nodes at level d (starting at d = 0)
is 2^d, with the time complexity of sifting down at level d is h - d. So:
Total work (W) = \sum_{d = 0}^h (2^d (h - d)), where h is log_2 {n}
W = log_2{n} + 2(log_2{n} - 1) + 2^2(log_2{n} - 2) + ... + 2^{log_2{n} - 2} (2) + 2^{log_2{n} - 1} 
2W =         + 2(log_2{n})     + 2^2(log_2{n} - 1) + ... + 2^{log_2{n} - 2} (3) + 2^{log_2{n} - 1} (2) + 2^{log_2{n}}       

Hence:
W = - log_2{n} + 2 + 2^2 + ... + 2^{log_2{n} - 1} + 2^{log_2{n}} = - log_2{n} + 2(2^{log_2{n}} - 1) = 2n - 2 - log_2{n} 
So we notice that the log_2{n} is bounded by the linear function n, and so we can conclude that
the time complexity is O(n) \qed

The space complexity here is just O(1) as we just store the node that we are at

Max heap: when the parent is greater than its children

A vital application of heaps is via putting sequences of values on the heap. 


"""

#Build a min heap: time O(n), space: O(1) we are just rearranging elements in space

A = [-4, 3,1, 0, 2, 5, 10,8, 12, 9]

import heapq
heapq.heapify(A)

print(A)

#Heap push: Time: O(log n)
heapq.heappush(A, 4)

#Heap.pop (Extract min), time to fix tree: O(log n)
min = heapq.heappop(A)

#heap sort, Time: O(n log n), space: O(n), note O(1) space is possible via swapping
# note as well that I didn't use a dynamic array (where I appended) as this would
# involve amortisation of O(1)*, but either way should work
def heapsort(arr):
    heapq.heapify(arr)
    n = len(arr)
    output = [0] * n
    for i in range(n):
        min = heapq.heappop(A)
        output[i] = min
        
    return output

print(A)

#Heap push pop: time O(log n)
heapq.heappushpop(A, 99) #pushes in 99, pops off 0

#peak at min: time O(1)
A[0]



#creating a max heap
B =  [-4, 3,1, 0, 2, 5, 10,8, 12, 9]
n = len(B)

for i in range(n):
    B[i] = -B[i]
    
heapq.heapify(B) 
for i in range(n):
    B[i] = -B[i]


print(B)
#so notice with max heaps, you must negate its sign

#building Heap from scratch (overtime)
C =  [-4, 3,1, 0, 2, 5, 10,8, 12, 9]
heap = []

for x in C:
    heapq.heappush(heap, x)
    print(heap)