"""
root is the beginning of a tree, trees are a directed graph, often from a child
you cannot get back to the parent

class Node:
    root.val 
    root.left
    root.right
leaves are nodes that don't have any other branches (both nulls), parent -> child

Complete/Full Tree: levels are mostly filled out (root must have both left and right children)
Perfect Tree: levels are all filled out

Can be expressed as an array as well:
root: 1
C_1 : 2
C_2 : 3
C_1,1 : 4
C_1,2 : 5
C_2,1 : 6
C_2,2 : 7
...
2i, 2i + 1
if null, no child
height/depth: number of levels


DEPTH FIRST SEARCH
-prioritise search by completing the depth of the tree. This is done by going all the way
down to the left, then once hitting the bottom of the tree and check that the node
has no children on the left and right, goes back up to its parent, check if there's a
child on the right, then continue as such

Preorder Traversal: first process the node we are currently at, then process its left child
, then process its right child
i.e

                1
        2               3
    4       5       10
    
is traversed [1, 2, 4, 5, 3, 10]

Inorder: first process left, then node, then right:
[4, 2, 5, 1, 10, 3]


Postorder: first process left, then right, then node:
[4, 5, 2, 3, 10, 1]

IMPLEMENTED USING A STACK (Recursive):
- create empty call stack and initialise it with the root node (as an object) inside
stack = [1]
while stack:
    #pop from the stack and process the node:(ask right first)
    
[3, 2] -pop-> [3], processed stack [1, 2] process the node with val 2 -> [3, 4, 5]
[3, 5, 4] -pop-> [3, 5], processed stack [1, 2, 4], process node 4 (no child)
[3, 5] -pop-> [3], processed stack [1, 2, 4, 5], processe node 5 (no child)
[3] -pop-> [], processed stack [1,2,4,5,3], process node 3 -> [10]
[10] -pop-> [], processed stack [1,2,4,5,3,10], process node 10 (no child)
(preorder traversal)


BREADTH FIRST SEARCH: level order traveral
[1, 2, 3, 4, 5, 10]
IMPLEMENTED USING A QUEUE
Q = [1]

while Q:
    #pop it off
    #if the node popped has a left child, put in Q, then right child, append in Q


[1] -pop-> [], processed stack [1], process node 1 -> [2, 3]
[2, 3] -pop-> [3] (pops from left for dequeue), processed stack [1, 2], process node 2 -> [3, 4, 5]
[3, 4, 5] -pop-> [4, 5], processed stack [1, 2, 3], processed node 3 -> [4, 5, 10]
[4, 5, 10] -pop-> ... -> [1, 2, 3, 4, 5, 10]



To look up for a value in a tree; time is O(n), space: O(n)
Abit more on the space complexity. From the tree, in the call stack what we actually see:
[1] -> [1,2] -> [1, 2, 4] -> [1, 2, 5] -> [1, 3] -> [1, 3, 10], so we actually have O(h)
which is the height of the tree. But in the worse case, h = n. This is also the case 
in BFS (via Queue), we always store the levels of the tree, i.e 1, 2, 4, 8, 16 etc. So 
in the worst case, the storing queue stores half of the tree in the lowest level, giving
O(n/2) -> O(n)

Binary search trees: are perfect trees such that for all node in the tree, the value is greater
than all nodes on the left, but less than all values on the right
. i.e:


                        5
            1                       8
      -1        3               7       9
      
A valuable property is when we lookup whether a value is in the tree, start at root, 
perform DFS. Say i.e we look for 9, we look at root node,  which has a value of 5, and so a 9
must occur to the right, cancelling the left. Then we look at node 8, noticing 
that 9 > 8 so we cancel out the left of 8. And then we see the 9. So the time complexity 
of this as we keep halving, is O(log n), assuming a height balanced tree, not lop-sided
as if all nodes are just to the right, its the same as a linked list
"""

#binary trees
class TreeNode:
    def __init__(self, val, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right
        
    def __str__(self):
        return (str(self.val))
    
#we use this tree:
"""
                1
        2               3
    4       5       10
"""
A = TreeNode(1)
B = TreeNode(2)
C = TreeNode(3)
D = TreeNode(4)
E = TreeNode(5)
F = TreeNode(10)

A.left = B
A.right = C
B.left = D
B.right = E
C.left = F

#Recursive Pre Order Traversal: DFS : Time: O(n), Space: O(n) 
#Node -> L-> R
def pre_order(node):
    if not node:
        return
    
    print(node)
    pre_order(node.left)
    pre_order(node.right)
    

def in_order(node):
    if not node:
        return
    
    in_order(node.left)
    print(node)
    in_order(node.right)
    
def post_order(node):
    if not node:
        return
    
    post_order(node.left)
    post_order(node.right)
    print(node)
    
#iterative pre-order traversal (DFS) via stack (you can only do pre-order, cause we
# aren't ding recursion)
def pre_order_iterative(node):
    stk = [node]
    
    while stk:
        node = stk.pop()
        print(node)
        if node.right:
            stk.append(node.right)
        #print(node)
        if node.left:
            stk.append(node.left)
        #print(node)
            
#pre_order(A)
#in_order(A)
#post_order(A)
#pre_order_iterative(A)

#level order traversal BFS:
from collections import deque
def level_order(node):
    #we implement a queue
    q = deque()
    q.append(node)
    
    while q:
        node = q.popleft()
        print(node)
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
            
#level_order(A)

#check if value exists (DFS):

#this code here does not work. It is important to know why. Notice, when we check
# for node.left or node.right, we don't perform a return, and so we run into the problem
# of the function returning none, which arises as we don't return in the current call
# and so it gets ignored. 
def check_val(node, target):
    if not node:
        return False
    
    if node.val == target:
        return True
    
    return check_val(node.left, target) or check_val(node.right, target)
    
    """
    if node.left and check_val(node.left, target):
        return True        
    
    if node.right and check_val(node.right, target):
        return True
    
    return False
    """

#print(check_val(A, 11))

# Binary Search Tree:
"""

                        5
            1                       8
      -1        3               7       9
"""
A2 = TreeNode(5)
B2 = TreeNode(1)
C2 = TreeNode(8)
D2 = TreeNode(-1)
E2 = TreeNode(3)
F2 = TreeNode(7)
G2 = TreeNode(9)

A2.left, A2.right = B2, C2
B2.left, B2.right = D2, E2
C2.left, C2.right = F2, G2

#time O(log n), space: O(log n) (assuming balanced tree)
def search(node, target):
    if not node:
        return False
    if node.val == target:
        return True
    if node.val < target:
        #we look at right only:
        return search(node.right, target)
    else:
        return search(node.left, target)
    
        
print(search(A2, 11))