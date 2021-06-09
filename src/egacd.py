from os import chroot
from numpy.lib.function_base import _calculate_shapes
from scipy.io import mmread
import numpy as np
from src.chromosome import Chromosome
from src.util import loadDataset

class EGACD():
    def __init__(self, path, populationsSize, pc, generation):
        self.mtx = loadDataset(path)
        self.populationSize = populationsSize
        self.chromosomeLen = self.mtx.shape[0]
        self.generation = generation
        self.pc = pc

    def initialization(self):
        population = [Chromosome(self.mtx) for _ in range(self.populationSize)]
        return population
    
    def crossover(self, parentA, parentB):
        prob = np.random.rand()
        if prob < self.pc:
            position = np.random.randint(1, self.chromosomeLen)

            # init chromosome (children)
            childrenA, childrenB = Chromosome(self.mtx), Chromosome(self.mtx)
            # one point crossover operation
            childrenA.chromosome = np.concatenate((parentA.chromosome[:position+1], parentB.chromosome[position+1:]))
            childrenB.chromosome = np.concatenate((parentB.chromosome[:position+1], parentA.chromosome[position+1:]))
            return childrenA, childrenB
        else:
            return parentA, parentB

    def mutation(self, parent):
        children = parent
        mutateGene = np.random.randint(self.chromosomeLen)
        neighbor = np.where(self.mtx[mutateGene]==1)[1]
        children.chromosome[mutateGene] = np.random.choice(neighbor)
        return children


    def operator(self, population):
        # populationChildren = np.zeros((self.populationSize, self.chromosomeLen))
        populationChildren = []
        for _ in range(int(self.populationSize/2)):
            prob = np.random.rand()
            candidate = np.random.randint(self.populationSize, size=2)
            # XO
            if prob < self.pc:
                childrenA, childrenB = self.crossover(population[candidate[0]],population[candidate[1]])
            # mutation
            else:
                childrenA, childrenB = self.mutation(population[candidate[0]]), self.mutation(population[candidate[1]])
            populationChildren.extend((childrenA, childrenB))

        # concatenate parent & offspring
        populationConcatenate = population + populationChildren
        return populationConcatenate
    
    def localSearch(self, population):
        for chromosome in population:
            chromosome.clusterize()
            chromosome.localSearch()
            chromosome.setModularity()
        return population

    def selection(self, population):
        selectedPopulation = sorted(population, key=lambda chromosome: chromosome.modularity, reverse=True)
        return selectedPopulation[:self.populationSize]

    def oneRun(self,population):
        populationDouble = self.operator(population)
        populationDouble = self.localSearch(populationDouble)
        # populationDouble = self.MBCrossover(populationDouble)
        populationSelected = self.selection(populationDouble)
        bestModularity = populationSelected[0].modularity
        return bestModularity, populationSelected

    def doIt(self):
        population = self.initialization()
        for _ in range(self.generation):
            bestModularity,population = self.oneRun(population)

        return bestModularity,population[0]


if __name__ == '__main__':
    path ='./soc-karate/soc-karate.mtx'
    mtx = loadDataset(path)
    egacd = EGACD(path, 50, 0.8, 100)

    mod_arr  = []
    repeat = 10
    time_arr = []

    for i in range(repeat):
        print("=== Start repeat [",i,"] ===")
        startTime = time.time()
        bestModularity, bestChromosome = egacd.doIt()
        print("Best Modularity: ",bestModularity)
        mod_arr.append(bestModularity)
        time_arr.append(time.time()-startTime)
        print("Time: ",time_arr[i])

    print("BEST:", max(mod_arr))
    print("AVG:", sum(mod_arr)/repeat)
    print("AVG DURATION:",sum(time_arr)/repeat)
