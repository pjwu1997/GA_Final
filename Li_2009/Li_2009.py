import random
import numpy as np
from GA import GA
import globals 
from scipy.io import mmread

def loadData(path):
	sparse = mmread(path)
	globals.AMartix = sparse.todense()
	globals.AMartix = np.array(globals.AMartix)
	globals.N_edge = sum(sum(globals.AMartix))/2

def main():
	globals.globals()
	f = "../soc-karate.mtx"
	loadData(f)

	ell = len(globals.AMartix)
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
			print("Modularity: ", result.getFittness())
			print("+")
		else: print("-, ",result.getFitness())

if __name__ == "__main__":
    main()