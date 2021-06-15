from os import chroot
import os
from numpy.lib.function_base import _calculate_shapes
from scipy.io import mmread
import numpy as np
from EGACD_MD.chromosome import Chromosome
from EGACD_MD.util import loadDataset, reducegraph, concateReduced, setModularity
import collections, copy
import time
import EGACD_MD.globals as globals
from multiprocessing import Pool

def writeProgress(pro_arr):
    f = open("progress.txt","w")
    for gen in pro_arr:
        f.write(str(gen[0])+'\t'+str(gen[1])+'\n')

class EGACD():
    def __init__(self, populationsSize, pc, generation, isParallel):
        self.reduced_mtx = globals.reduced_mtx
        self.populationSize = populationsSize
        self.chromosomeLen = self.reduced_mtx.shape[0]
        self.generation = generation
        self.isParallel = isParallel
        self.pc = pc
        self.nfe = 0

    def initialization(self):
        self.nfe = 0
        population = [Chromosome() for _ in range(self.populationSize)]
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
            childrenA, childrenB = Chromosome(), Chromosome()
            # one point crossover operation
            childrenA.chromosome = np.concatenate((parentA.chromosome[:position+1], parentB.chromosome[position+1:]))
            childrenB.chromosome = np.concatenate((parentB.chromosome[:position+1], parentA.chromosome[position+1:]))
            return childrenA, childrenB
        else:
            return parentA, parentB

    def mutation(self, parent):
        children = parent
        mutateGene = np.random.randint(self.chromosomeLen)
        neighbor = np.where(self.reduced_mtx[mutateGene]==1)[1]
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
    
    def localSearch_p(self, chromosome):
        if not chromosome.evaluated:
            chromosome.clusterize()
        chromosome.localSearch()
        chromosome.setModularity()
        return chromosome

    def selection(self, population,size):
        selectedPopulation = sorted(population, key=lambda chromosome: chromosome.modularity, reverse=True)
        return selectedPopulation[:size]

    def oneRun(self,population):
        populationDouble = self.operator(population)
        if self.isParallel:
            populationDouble = self.MBCrossover_p(populationDouble)
        else:
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
        self.nfe_mod_arr = []
        for _ in range(self.generation):
            bestModularity,population = self.oneRun(population)
            self.nfe_mod_arr.append([self.nfe,bestModularity])

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

    def MBCrossover_p(self, population):
        for chrom in population:
            if not chrom.evaluated:
                chrom.clusterize()
                chrom.setModularity()
                self.nfe += 1

        populationSelected = self.selection(population,int(self.populationSize/2))
        linkageDictionary = self.linkageIdentify(populationSelected)

        BB = self.findBB(linkageDictionary)

        obj = [(pop, BB) for pop in population]
        population = globals.pool.starmap(self.MBCsingle, obj)
        self.nfe += len(population)
        return population

    def MBCsingle(self, chromosome, BB):
        tmp = copy.deepcopy(chromosome)
        for gene in range(len(BB) - 1):
            tmp.chromosome[BB[gene]] = BB[gene+1]
        tmp.clusterize()
        tmp.setModularity()
        if tmp.modularity > chromosome.modularity:
            return tmp
        else:
            return chromosome

if __name__ == '__main__':
    path ='../soc-karate/soc-karate.mtx'
    mtx = loadDataset(path)
    isReduced = False
    isParallel = True
    cpu_count = os.cpu_count()
    num_workers = 5
    globals.pool = Pool(num_workers)

    if isReduced:
        obj = reducegraph(path, 0.2)
        globals.index_selected, globals.index_eliminated, globals.mtx, globals.reduced_mtx = \
            obj['index_selected'], obj['index_eliminated'], obj['original_mtx'], obj['reduced_mtx']
        globals.edge = np.count_nonzero(mtx==1) / 2
        egacd = EGACD(30, 0.8, 50, isParallel)
    else:
        globals.reduced_mtx = loadDataset(path)
        egacd = EGACD(30, 0.8, 50, isParallel)

    mod_arr  = []
    repeat = 5
    time_arr = []
    nfe_arr = []
    cluster_arr = []
    pro_arr = []
    for i in range(repeat):
        print("=== Start repeat [",i,"] ===")
        print('========Number of cpu: ' + str(cpu_count) + '===========')
        print('Use' + str(num_workers) + ' cores.')

        startTime = time.time()
        bestModularity, bestChromosome = egacd.doIt()
        if isReduced:
            concateReduced(bestChromosome)
            setModularity(bestChromosome, globals.edge)
            bestModularity = bestChromosome.modularity
        print("Best Modularity: ", bestModularity)
        print("Best Cluster: ", bestChromosome.cluster)
        cluster_arr.append([int(i) for i in bestChromosome.cluster])
        mod_arr.append(bestModularity)
        pro_arr.append(egacd.nfe_mod_arr)

        time_arr.append(time.time()-startTime)
        print("Time: ",time_arr[i])

        nfe_arr.append(egacd.nfe)
        print("NFE: ", egacd.nfe)

    max_mod = max(mod_arr)
    max_index = mod_arr.index(max_mod)
    max_pro = pro_arr[max_index]
    print("BEST:", max(mod_arr))
    print("AVG:", sum(mod_arr)/repeat)
    print("AVG DURATION:",sum(time_arr)/repeat)
    print("AVG NFE:", sum(nfe_arr)/repeat)
    np.save('cluster.npy',{'cluster': cluster_arr[max_index], 'mod': max_mod})
    if not isReduced:
        writeProgress(max_pro)

