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

		# while len(np.unique(genes)) < N_module:
		# 	genes = []
		# 	for i in range(ell):
		# 		genes += [np.random.randint(0,self.N_module)]

		self.genes = np.array(genes)

		self.evaluated = False
		self.fitness = 0

	def setVal(self,index,val):
		if(self.genes[index] == val): return

		self.genes[index] = val
		self.evaluated = False

	def getVal(self,index):
		return self.genes[index]

	def getFitness(self):
		if not self.evaluated:
			self.fitness = self.modularity()
			self.evaluated = True
		return self.fitness
		
	def modularity(self):
		e_pp = np.zeros(self.N_module)
		e_pq = np.zeros(self.N_module)
		for p in range(self.ell):
			p_module = self.genes[p]
			for q in range(self.ell):
				if p_module == self.genes[q]:
					e_pp[p_module] += globals.AMartix[p][q]
				else: 
					e_pq[p_module] += globals.AMartix[p][q]
		e_pp /= 2
		e_pq += e_pp
		# print(e_pp,e_pq)

		e_pp /= globals.N_edge
		e_pq /= globals.N_edge*2

		Q_value = sum(e_pp - e_pq**2)
		return Q_value
	