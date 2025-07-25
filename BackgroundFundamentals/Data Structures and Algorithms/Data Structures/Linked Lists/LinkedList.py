#Singly linked list, nodes are connected in a single direction 1 -> 2 -> 3 ->
# Linked list node is an object that has a memory address (not like an array which is
# a continguous sequence of memory). 3 here is directed to null (None) which is a marker
# specifying the end of a list
"""
class Node:
    Node.value
    Node.next (next address)
"""
#to insert a value at a node, you must traverse the list from the head, O(n), adding/
# deleting something at the beginning of the list is O(1), a flaw with a singly linked
# list is that if you only have access to the reference to the node in the list 
# that is not the header and want to delete the node, you cant change where the preceeding node
# points to.

#doubly linked list, allows a reverse direction to all nodes, where the conventional 
#head points to none as well.
"""
class Node:
    Node.value
    Node.next (next address), to another node object
    Node.prev
"""
#singly linked list

class SinglyNode:
    def __init__(self, val, next = None):
        self.val = val
        self.next = next
    def __str__(self):
        return(str(self.val))


Head = SinglyNode(1)
A = SinglyNode(3)
B = SinglyNode(4)
C = SinglyNode(7)

#connect these up
Head.next = A
A.next = B
B.next = C

print(Head)

#Traverse the list - O(n)
curr = Head
while curr:
    print(curr)
    curr = curr.next
    
#Display linked list
def display(head):
    curr = head
    elements = []
    while curr:
        elements.append(str(curr.val))
        curr = curr.next
    print(' -> '.join(elements))
    #['a', 'b', 'c'] turns to abc via join
    #turns to a -> b -> c-> via ' -> '.join(elements)

display(Head)

#search for node value
def search(head, val):
    curr = head
    while curr:
        if val == curr.val:
            return True
        curr = curr.next
    return False

search(Head, 2)

#doubly linked list

class DoublyNode:
    def __init__(self, val, next = None, prev = None):
        self.val = val
        self.next = next
        self.prev = prev
    def __str__(self):
        return (str(self.val))
    
head = tail = DoublyNode(1)

def displayDouble(head):
    curr = head
    elements = []
    while curr:
        elements.append(str(curr.val))
        curr = curr.next
    print('<->'.join(elements))

#insert at beginning
def insert_at_beginning(head, tail, val):
    new_node = DoublyNode(val, next = head)
    head.prev = new_node
    return new_node, tail

head, tail = insert_at_beginning(head, tail, 3)
displayDouble(head)

#insert at end
def insert_at_end(head, tail, val):
    new_node = DoublyNode(val, prev = tail)
    tail.next = new_node
    return head, new_node

head, tail = insert_at_end(head, tail, 4)
displayDouble(head)