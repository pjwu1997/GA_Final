from os import chroot
from numpy.lib.function_base import _calculate_shapes
from scipy.io import mmread
import numpy as np
from src.chromosome import Chromosome
from src.util import loadDataset
import collections, copy
import time
import src.globals

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
    
    def linkageIdentify(self,population):
        populationMatrix = []
        for chrom in population:
            populationMatrix.append(chrom.chromosome)

        populationMatrixTrans = np.array(populationMatrix).transpose()

        dicArray = []
        for i in range(self.chromosomeLen):
            dicArray.append(collections.Counter(populationMatrixTrans[i]))

        return dicArray

    def findBB(self,linkage):
        startNode = np.random.randint(0,self.chromosomeLen)
        BB = []
        nextNode = startNode

        while nextNode not in BB:
            BB.append(nextNode)
            nextNode = linkage[nextNode].most_common(1)[0][0]

        BB.append(nextNode)
        return np.array(BB)


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
        children.evaluated = False
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
            if not chromosome.evaluated:
                chromosome.clusterize()
            chromosome.localSearch()
            chromosome.setModularity()
        return population

    def selection(self, population,size):
        selectedPopulation = sorted(population, key=lambda chromosome: chromosome.modularity, reverse=True)
        return selectedPopulation[:size]

    def oneRun(self,population):
        populationDouble = self.operator(population)
        populationDouble = self.MBCrossover(populationDouble)
        # populationDouble = self.localSearch(populationDouble)
        populationSelected = self.selection(populationDouble,self.populationSize)
        bestModularity = populationSelected[0].modularity
        return bestModularity, populationSelected

    def oneRunMB(self,population):
        population = self.MBCrossover(population)
        population = self.localSearch(population)
        bestModularity = population[0].modularity
        return bestModularity, population

    def doIt(self):
        population = self.initialization()
        for _ in range(self.generation):
            bestModularity,population = self.oneRun(population)

        return bestModularity,population[0]

    def MBCrossover(self,population):

        for chrom in population:
            if not chrom.evaluated:
                chrom.clusterize()
                chrom.setModularity()

        populationSelected = self.selection(population,int(self.populationSize/2))
        linkageDictionary = self.linkageIdentify(populationSelected)

        BB = self.findBB(linkageDictionary)

        for i in range(len(population)):
            tmp = copy.deepcopy(population[i])
            for gene in range(len(BB)-1):
                tmp.chromosome[BB[gene]] = BB[gene+1]
            tmp.clusterize()
            tmp.setModularity()

            if tmp.modularity > population[i].modularity:
                population[i] = tmp
        return population


if __name__ == '__main__':
    path ='./soc-karate/soc-karate.mtx'
    mtx = loadDataset(path)
    egacd = EGACD(path, 50, 0.8, 100)

    mod_arr  = []
    repeat = 10
    time_arr = []
    nfe_arr = []
    for i in range(repeat):
        src.globals.nfe = 0
        print("=== Start repeat [",i,"] ===")

        startTime = time.time()
        bestModularity, bestChromosome = egacd.doIt()

        print("Best Modularity: ",bestModularity)
        mod_arr.append(bestModularity)

        time_arr.append(time.time()-startTime)
        print("Time: ",time_arr[i])

        nfe_arr.append(src.globals.nfe)
        print("NFE: ",src.globals.nfe)

    print("BEST:", max(mod_arr))
    print("AVG:", sum(mod_arr)/repeat)
    print("AVG DURATION:",sum(time_arr)/repeat)
    print("AVG NFE:", sum(nfe_arr)/repeat)
    

