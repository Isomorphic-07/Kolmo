"""
2's comp: flip all digits and add 1. Convert negative to pisitive, flip all bits then add 1
airhtmetic bit shift keeps the sign bit, while logical bit shift we do change (often for right shifts)
"""

#decimal to binary
print(bin(5)[2:]) #usually would print out 0b101, [2:] cuts this out

#bin to dec
binary_5 = '101'
int(binary_5, 2) #2 tells us the base

#hexadec

hex(25)

#Arithmetic right shift (signed):
5 >> 1

#left shift
5 << 2