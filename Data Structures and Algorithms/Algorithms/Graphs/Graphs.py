#cyclic graphs are graphs that contains a cycle in it
#edge list: a list of edges shown by 2 vertices of that edge
#adjacency matrix (nodes and connection, very similar to transition matrices in markov chains)
#adjacency list: a hash map where the key is the vertex value and the value is a list of 
#its neighbours
"""
class Node:
    node.value
    node.neighbours = [] (references to the nodes of neighbours)
    
    
DFS Traversal of Graphs:
It is very important to have a "seen" set so that we know not to traverse on
a node that we have already visited. Vital so that we ignore cycles

Recursive: essentially start at a vertex and pick any edge, and travel. Once we get
to a point where there is nowhere to go or all neighbours are in the seen set, we back track
(remember, we add nodes we visit into the seen set), uses recursive call stack

Iterative:
initialise on the stack with a vertex source,
while stack:
    stack.pop()
    add into the stack the connections of the popped node
    (remember add the visited nodes into the seen stack)
    (anything added to the stack will be added to the seen set)
    
    
    
BFS: (via queue)
initialise queue with source node, pop on the left allows us to see in the breadth order

Complexity: assuming we use an adjacency list

Time: you will see all the vertices and all edges: O(V + E)
Space: O(V+ E)


Trees: connected acyclic graph (connected: you can get anywhere from anywhere)
Trees: connected and acyclic graphs. Given V the number of vertices of a tree, 
the number of edges E = V - 1.
Proof: We can prove this by induction:
Base case:
    V = 1, E = 0
Inductive:
    V = k, E = k-1
    V = k +1, this is a tree of k vertices and the k+1 vertex is initially unconnected. 
    Now, to create a tree, we must be connected
    so we can connect this outer node to any other vertex in the k-tree. Since 
    the k-tree is already connected, this new path allows
    us to be connected as well. So we know that we must have at least k edges, 
    but could we have more? Notice if we now add another edge anywhere
    in the tree, this creates cycle somewhere as a node that is connected to 
    this new edge now has more than one way to get
    to another node creating a cycle. Thus, for V = k + 1, E = k
"""
#array of edges (Directed): [start, end]
n = 8
A = [[0, 1], [1, 2], [0, 3], [3, 4], [3, 6], [3, 7], [4, 2], [4, 5], [5, 2]]

#Convert array of edges to an adjacency matrix

M = []
for i in range(n):
    M.append([0] * n) #creates a matrix nxn

for u, v in A:
    M[u][v] = 1 #populates adjacency matrix
    #if we want an undirected graph, also write M[v][u] = 1 (Symmetric matrix)


#Array of edges to adjacency list
from collections import defaultdict

D = defaultdict(list)
for u,v in A:
    D[u].append(v)
    #undirectied: D[v].append(u)

#DFS with Recursion: O(V + E)
def dfs_recursive(node):
    print(node) #process
    for nei_node in D[node]:
        if nei_node not in seen:
            seen.add(nei_node)
            dfs_recursive(nei_node)
    
source = 0

seen = set()
seen.add(source)
dfs_recursive(source)

#Iterative DFS with stack:

source = 0
seen = set()
seen.add(source)
stack = [source]

while stack:
    node = stack.pop()
    print(node)
    for nei_node in D[node]:
        if nei_node not in seen:
            seen.add(nei_node)
            stack.append(nei_node)
 
#BFS via queue
source = 0

from collections import deque
seen = set()
seen.add(source)
q = deque()
q.append(source)

while q:
    node = q.popleft()
    print(node)
    for nei_node in D[node]:
        if nei_node not in seen:
            seen.add(nei_node)
            q.append(nei_node)
            
#Graphs as classes
            
class Node:
    def __init__(self, value):
        self.value = value
        self.neighbours = []
        
    def __str__(self):
        return f'Node({self.value})'
    
    def display(self):
        connections = [node.value for node in self.neighbours]
        return f'{self.value} is connected to: {connections}'
    
A = Node("A")
B = Node("B")
C = Node("C")
D = Node("D")

A.neighbours.append(B)
B.neighbours.append(A)

C.neighbours.append(D)
D.neighbours.append(C)

print(A.display())