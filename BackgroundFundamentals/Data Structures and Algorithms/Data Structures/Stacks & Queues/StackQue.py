#stack LIFO (implemented via dynamic arrays)
#queue FIFO, annd so dequeing something (for queues we remove the first thing) in O(n)
# as this involved moving every element back from removing the first element. However
# via a doubly linked list, dequeue is O(1) as it just involved removing the header node
# and move the reference point to the 2nd element (implemented via doubly linked lists)


#stacks

stk = []
stk.append(5)
stk.append(4)
stk.append(3)

x = stk.pop()
print(x)
print(stk)

#Queues- FIFO
from collections import deque

q = deque() #operates as a double ended queue
print(q)

#enqueue- add element to the right
q.append(5)
q.append(6)
print(q)

#deque (pop left)
q.popleft()
print(q[0])
print(q[-1])