#Recursive fucntions use a function call stack (as we repeatedly call the function in recursions)
# the reason why this is needed is because when you re-call a function, you store it
# into a memory address in the fucntion call stack, which is linked to the address to 
# another function call above in the stack

#notice that for Fibonnaici for example:
"""
f(5) = f(4) + f(3) = [f(3) + f(2)] + [f(2) + f(1)] = [[f(2)+ f(1)] + f(1) + f(0)] + [[f(1) + f(0)] + f(1)]...
which is time complexiy O(2^n), so doubling effect occurs as at each call of the function 
you sprout into 2 function calls. However, the space capacity, since we are constantly
storing the function call into the stack all at the same time once, and so it is O(n)
"""

#time O(2^n), space O(n)
def F(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return F(n-1) + F(n-2)
    
    
class SinglyNode:
    def __init__(self, val, next = None):
        self.val = val
        self.next = next
    def __str__(self):
        return(str(self.val))
    
Head = SinglyNode(1)
a = SinglyNode(3)
b = SinglyNode(4)
c = SinglyNode(7)

Head.next = a
a.next = b
b.next = c

#linked lists, Time: O(n), Space O(n)
def reverse(node):
    if not node:
        return
    reverse(node.next)
    print(node)

reverse(Head)        