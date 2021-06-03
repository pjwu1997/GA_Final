from Chromosome import Chromosome
import globals
from scipy.io import mmread
import numpy as np


def loadData(path):
	sparse = mmread(path)
	globals.AMartix = sparse.todense()
	globals.AMartix = np.array(globals.AMartix)
	globals.N_edge = sum(sum(globals.AMartix))/2

globals.globals()
f = "../soc-karate.mtx"
loadData(f)

test = Chromosome(34,4)
best_arr = [3,3,3,3,4,4,4,3,2,2,4,3,3,3,2,2,4,3,2,3,2,3,2,1,1,1,2,1,1,2,2,1,2,2]
for i in range(34):
	test.setVal(i,best_arr[i]-1)

print(test.getFitness())
