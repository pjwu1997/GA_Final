import random
import numpy as np
import globals

class Chromosome:
	def __init__(self,ell,N_module):
		self.ell = ell
		self.N_module = N_module
		genes = []
		for i in range(ell):
			genes += [np.random.randint(0,self.N_module)]

		while len(np.unique(genes)) < N_module:
			genes = []
			for i in range(ell):
				genes += [np.random.randint(0,self.N_module)]

		self.genes = np.array(genes)

		self.evaluated = False
		self.fitness = 0

	def setVal(self,index,val):
		if(self.genes[index] == val): return

		self.genes[index] = val
		self.evaluated = False

	def getVal(self,index):
		return self.genes[index]

	def getFittness(self):
		if not self.evaluated:
			self.fitness = self.modularity()
			self.evaluated = True
		return self.fitness
		
	def modularity(self):
		e_pp = np.zeros(self.N_module)
		e_pq = np.zeros(self.N_module)
		for p in range(self.ell):
			p_module = self.genes[p]
			for q in range(p+1,self.ell):
				if p_module == self.genes[q]:
					e_pp[p_module] += globals.AMartix[p][q]/(globals.N_edge)
				else:
					e_pq[p_module] += globals.AMartix[p][q]/(globals.N_edge)

		Q_value = sum(e_pp) - sum(e_pq**2)
		return Q_value


	