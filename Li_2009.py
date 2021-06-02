import random
import numpy as np
from GA import GA
import globals 
from scipy.io import mmread

def main():
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

	ell = N_node
	N_module = 4
	nInitial = 20
	pc = 0.8
	pm = 0.1
	lsN = 4
	maxGen = 3000
	maxFGen = 100

	repeat = 1
	bestQ = 0.401

	ga = GA(ell,N_module,nInitial,pc,pm,lsN,maxGen,maxFGen)

	for i in range(repeat):
		ga.initPoluation()
		result = ga.run()
		if result.getFittness() >= bestQ:
			print(result.genes)
			print("+")
		else: print("-, ",result.getFitness())

if __name__ == "__main__":
    main()