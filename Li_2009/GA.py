import random
import numpy as np
from Chromosome import Chromosome
import copy
import globals

class GA:

	def __init__(self,ell,N_module,nInitial,pc,pm,lsN,maxGen,maxFGen):
		self.ell = ell
		self.nCurrent = nInitial
		self.N_module = N_module
		self.pc = pc
		self.pm = pm
		self.lsN = lsN
		self.maxGen = maxGen
		self.maxFGen = maxFGen

		self.population = [Chromosome(ell,N_module) for i in range(self.nCurrent)]

	def initPoluation(self):
		self.population = [Chromosome(self.ell,self.N_module) for i in range(self.nCurrent)]


	def tournamentSelection(self,p):
		N_array = np.random.permutation(self.nCurrent)
		for i in range(p-1):
			N_array = np.append(N_array,np.random.permutation(self.nCurrent))

		selected_index = []
		
		i = 0
		while i < self.nCurrent*p:
			winner = i
			winner_fittness = self.population[i].getFittness()
			for j in range(1,p):
				rival_fitness = self.population[i+j].getFittness()
				if winner_fittness < rival_fitness:
					winner_fittness = rival_fitness
					winner = i + j
			selected_index.append(winner)
			i += p


	def RWSelection(self):
	  
	    # Computes the totallity of the population fitness
	    population_fitness = [chromosome.getFittness() for chromosome in self.population]
	    f_min = min(population_fitness)

	    if f_min < 0: population_fitness = [i-f_min for i in population_fitness]
	    f_sum = sum(population_fitness)

	    # Computes for each chromosome the probability 
	    chromosome_probabilities = [f/f_sum for f in population_fitness]
	    
	    N_array = [i for i in range(self.nCurrent)]

	    # Selects one chromosome based on the computed probabilities
	    selectIndex = np.random.choice(N_array,size=(self.nCurrent*2),p=chromosome_probabilities)

	    newPopulation = []
	    for n in selectIndex:
	    	newPopulation.append(copy.deepcopy(self.population[n]))

	    self.population = newPopulation


	def checkValid(self,chromo):
		uni_module = [chromo.getVal(k) for k in range(self.ell)]
		uni_module = np.unique(uni_module)
		return len(uni_module) == self.N_module

	def crossover_ver1(self):
		newPopulation = [Chromosome(ell) for i in range(nCurrent)]

		for n in range(self.nCurrent):

			i = np.random.randint(0,nCurrent)
			j = np.random.randint(0,nCurrent)
			while i != j:
				j = np.random.randint(0,nCurrent)

			r = np.random.randint(0,self.ell)
			r_module = self.population[i].getVal(r)

			for k in range(self.ell):
				if self.population[i].getVal(k) == r_module:
					newPopulation[n].setVal(k,r_module)
				else:
					newPopulation[n].setVal(k,self.population[j].getVal(k))

		self.population = newPopulation

	def crossover(self):

		newPopulation = []
		n = 0
		while n < self.nCurrent*2:
			i = n
			j = n+1

			if np.random.rand() < self.pc:
				r = np.random.randint(0,self.ell)
				r_module = self.population[i].getVal(r)

				copyChromo = copy.deepcopy(self.population[j])
				for k in range(self.ell):
					if self.population[i].getVal(k) == r_module:
						copyChromo.setVal(k,r_module)

				uni_module = [copyChromo.getVal(k) for k in range(self.ell)]
				uni_module = np.unique(uni_module)

				count = 0
				while not self.checkValid(copyChromo) and count < 10:
					r = np.random.randint(0,self.ell)
					r_module = self.population[i].getVal(r)

					copyChromo = copy.deepcopy(self.population[j])
					for k in range(self.ell):
						if self.population[i].getVal(k) == r_module:
							copyChromo.setVal(k,r_module)
					count += 1

				if count >= 10:
					newPopulation.append(self.population[j])
				else: newPopulation.append(copyChromo)

			else: newPopulation.append(self.population[j])

			n += 2

		self.population = newPopulation


	def mutation(self):
		for n in range(self.nCurrent):
			if np.random.rand() < self.pm:
				i = np.random.randint(0,self.ell)
				j = np.random.randint(0,self.ell)
				gene_i = self.population[n].getVal(i)
				while i != j and gene_i != self.population[n].getVal(j):
					j = np.random.randint(0,self.ell)

				self.population[n].setVal(j,gene_i)

	def localSearch(self,chromosome):
		bestChromo = chromosome
		for n in range(self.lsN):
			newChromo = copy.deepcopy(chromosome)
			j = np.random.randint(0,self.ell)

			for q in range(self.ell):
				if globals.AMartix[j][q]:
					newChromo.setVal(q,chromosome.getVal(j))
			count = 0 
			while not self.checkValid(newChromo) and count < 10:
				newChromo = copy.deepcopy(chromosome)
				j = np.random.randint(0,self.ell)
				for q in range(self.ell):
					if globals.AMartix[j][q]:
						newChromo.setVal(q,chromosome.getVal(j))

			if count >= 10: continue

			if newChromo.getFittness() >= bestChromo.getFittness():
				bestChromo = newChromo

		return bestChromo


	def oneRun(self,prevBest):

		self.RWSelection()
		self.crossover()
		self.mutation()

		for i in range(self.nCurrent):
			self.population[i] = self.localSearch(self.population[i])

		bestIndex = 0
		bestFitness = self.population[bestIndex].getFittness()
		for i in range(self.nCurrent):
			if self.population[i].getFittness() > bestFitness:
				bestIndex = i
				bestFitness = self.population[i].getFittness()
		
		if bestFitness < prevBest.getFittness():
			self.population[bestIndex] = prevBest
			return prevBest

		return copy.deepcopy(self.population[bestIndex])

	def run(self):

		count = 0
		bestChromo = Chromosome(self.ell, self.N_module)
		bestFitness = bestChromo.getFittness()

		for i in range(self.maxGen):
			bestChromo = self.oneRun(bestChromo)
			if bestChromo.getFittness() <= bestFitness:
				count += 1
			else : count = 0 
			bestFitness = bestChromo.getFittness()
			
			if count > self.maxFGen:
				return bestChromo
			# self.printPopulation()

		return bestChromo


	def printPopulation(self):
		for i in self.population:
			print(i.genes)



