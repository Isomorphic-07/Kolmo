#Hash tables are comprised of 'buckets' which takes in an input, processes it 
# through a Hash function and outputs an index. A collision can occur when multiple inputs
# could be mapped to the same bucket. This is where we use seperate chaining (via linked lists)
# where we store the inputs mapped to the same bucket in a linked list, and so it
# also allows use to keep adding on to the chain. So why is Hashing good? Say
# for example we want to check if an input is in the hash table/set, this can be
# done in O(1)* amortised time (relative to N = number of elements in hash set) as we can just
# compute the hash function on that input and lookup its corresponding index. (assuming a good hash function)

#Set: collection of unique items, adding into hash set is O(1) (adding to the head)
# remove: O(1)*

#Maps: all functionality of a set, but can store data, key:value, keys must be hashable
# so now you have a linked list where the elements have multiple things in it (could a tuple for example)

#Linear Probing: (a technique ot handle collisions), instead of chaining, when an input
# has the same index as another input, it will be allocated to the next available index (+1)
# So for lookup, compute the index via the hash function, check the value in the index 
# in the hash set/map, if its not the value we're looking for look at the next index
# etc. We would return false if we hit a blank in an index. O(1)*

#however, if we decide to delete a term in a bucket, and that term has the same index value (via hashing)
# to the value where are trying to lookup, this can cause issues as the search would instantly see a blank
# in the index. So a way to fix this is to leave a -1 in the bucket when you delete something
# so that it is not null and returns false when looking up

#What is HASHABLE?: strings, integers, tuples (frozen array)(immutable)
#not Hashable: Arrays (dynamic arrays), lists, dictionaries (mutable)
# reason for this is that we want to maintain consistency in hashtable

#Hash set
s = set()

s.add(1)
s.add(2)
s.add(3)

print(s)

if 1 in s:
    print(True)

#s.remove(4) #throws key error
s.remove(3)

string = 'jskjsifjiejfijslpwpwp'
sett = set(string) # O(s), s length of string, gives all unique elements

print(sett)

#Hashmaps: dictionaries
d = {'greg':1, 'steve': 2, 'rob': 3}

#add
d['arsh'] = 4

if 'greg' in d:
    print(True)
    
#check value corresponding to key in dict: O(1)

#loop over the key:val pairs of dict O(n)
for key,val in d.items():
    print(f'key {key} : val {val}')
    
#default dict
from collections import defaultdict

default = defaultdict(int) #establishes a deault dictionary where the keys are
#int type

print(default[2]) #does not throw key error 

print(default)

#counter
from collections import Counter
counter = Counter(string)

print(counter) #gives a dictionary where the keys are characters in string and values
#are the number of times the character shows up in string