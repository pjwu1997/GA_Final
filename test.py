from Chromosome import Chromosome 
import globals
import numpy as np

globals.globals()
f = open("out.ucidata-zachary")

f.readline()
parm_array = f.readline().split()
globals.N_edge = int(parm_array[1])
N_node = int(parm_array[2])

print(globals.N_edge,N_node)
globals.AMartix = np.zeros([N_node,N_node])
for i in range(globals.N_edge):
	arr = f.readline().split()
	globals.AMartix[int(arr[0])-1][int(arr[1])-1] = 1
	globals.AMartix[int(arr[1])-1][int(arr[0])-1] = 1

test = Chromosome(34,4)
print(test.getFittness())

