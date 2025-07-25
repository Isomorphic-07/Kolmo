#what is usually done in practice

#Time complexity is O(n log n) from using Tim Sort 

G = [-5, 3, 2, 1, -3, -3, 7, 2, 2]

#in place, constant space O(1)
G.sort()

#get a new sorted array - O(n) space
H = [-5, 3, 2, 1, -3, -3, 7, 2, 2]
sorted_H = sorted(H)

#Sort array of tuples
I = [(-5, 3), (2, 1), (-3, -3), (7, 2), (2, 2)] #Intervals

sorted_I = sorted(I, key = lambda t: t[0]) #first positions are ascending

sorted_I = sorted(I, key = lambda t: -t[1]) #second positions in descending 