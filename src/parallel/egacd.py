import numpy as np
from scipy import stats
from parallel.util import loadDataset
import os
import time
from scipy.io import mmread
import copy
from scipy import stats

from multiprocessing import Pool
import itertools

global mtx
global reduced_mtx
global index_selected
global index_eliminated
global edge
global nfe
global pool

def loadDataset(path):
    mtx = mmread(path)
    mtx = mtx.todense()
    return mtx

def reducegraph(path, percentage):
    """
    Reduce the size of adjacency matrix by percentage.
    """
    mtx = mmread(path)
    m = mtx.tocsr()
    num_nodes = m.shape[0]
    cut_off_size = int(percentage * num_nodes)
    # print(m.shape[0]) # show number of nodes
    # print(m[0].nonzero()[1]) # show indices of non-zero elements
    neighbor_num_list = []
    for node_num in range(num_nodes):
        num_neighbor = len(m[node_num].nonzero()[1])
        neighbor_num_list.append((node_num, num_neighbor))
    # sort the list and cut-off
    neighbor_num_list.sort(key=lambda x: x[1])
    index_list = sorted([x[0] for x in neighbor_num_list[cut_off_size:]]) # rearrange from low to high
    print(set(index_list))
    index_cut_list = sorted(list(set(np.arange(num_nodes)) - set(index_list)))
    sub_mtx = m[index_list, :][:, index_list]
    return {
        'index_selected': index_list,
        'index_eliminated': index_cut_list,
        'original_mtx': m.todense(),
        'reduced_mtx': sub_mtx.todense()
    }

def writeProgress(pro_arr):
    f = open("progress.txt","w")
    for gen in pro_arr:
        f.write(str(gen[0])+'\t'+str(gen[1])+'\n')

def concateReduced(Chromosome):
    """
    Construct full_chromosome and full_cluster.
    """
    full_chromosome = np.zeros(mtx.shape[0])
    full_cluster = np.zeros(mtx.shape[0])
    copy_chromosome = copy.deepcopy(Chromosome.chromosome)
    for index, neighbor in enumerate(copy_chromosome):
        copy_chromosome[index] = index_selected[neighbor]
    full_chromosome[index_selected] = copy_chromosome
    clusterNum = Chromosome.clusterNum
    # 看連哪個cluster最多
    # 選一個連
    # 轉成對應index, cluster
    full_cluster[index_selected] = Chromosome.cluster
    # find the nodes connected to outsideGene
    for outsideGene in index_eliminated:
        # print('outside: ',np.array(self.full_mtx[outsideGene])[0].shape)
        neighbors = np.where(np.array(mtx[outsideGene])[0][index_selected]==1)
        if not neighbors:
            full_cluster[outsideGene] = clusterNum + 1
            clusterNum += 1
        else:
            neighbors = neighbors[0]
        for index, neighbor in enumerate(neighbors):
            neighbors[index] = index_selected[neighbor]
            # find which cluster the mutateGene connects to the most
            changeNum = stats.mode(full_cluster[neighbors])[0][0]
            fitNeighbor = np.where(full_cluster==changeNum)[0]
            full_cluster[outsideGene] = changeNum
            full_chromosome[outsideGene] = np.random.choice(fitNeighbor)
    Chromosome.chromosome, Chromosome.cluster, Chromosome.cluserNum = full_chromosome, full_cluster, clusterNum

def setModularity(Chromosome, edge):
    Qvalue = 0
    chromosomeLen = len(Chromosome.chromosome)
    for c in range(1, Chromosome.clusterNum+1):
        inValue = 0
        outValue = 0
        for i in range(chromosomeLen):
            for j in range(i + 1, chromosomeLen):
                if Chromosome.cluster[i] == c and Chromosome.cluster[j] == c:
                    inValue += mtx[i, j]
                    # outValue += self.mtx[i, j]
                elif Chromosome.cluster[i] != c and Chromosome.cluster[j] == c:
                    outValue += mtx[i, j]/2
                elif Chromosome.cluster[j] != c and Chromosome.cluster[i] == c:
                    outValue += mtx[i, j]/2
        outValue += inValue
        Qvalue += inValue / edge
        Qvalue -= (outValue / edge)**2
    Chromosome.modularity = Qvalue
class Chromosome():
    def __init__(self):
        self.reduced_mtx = reduced_mtx
        self.edge = np.count_nonzero(self.reduced_mtx == 1) / 2
        self.chromosomeLen = self.reduced_mtx.shape[0]
        # print(self.edge)
        self.cluster = np.zeros(self.chromosomeLen, dtype='int')
        self.clusterNum = 1

        chromosome = np.zeros((self.chromosomeLen), dtype='int')
        for gene in range(self.chromosomeLen):
            neighbor = np.where(self.reduced_mtx[gene]==1)[1]
            chromosome[gene] = np.random.choice(neighbor)
        self.chromosome = chromosome
        self.modularity = None

    def __str__(self):
        return str(self.chromosome)

    def clusterize(self):
        for gene in range(self.chromosomeLen):
            if self.cluster[gene] == 0:
                neighbors = self.__getNeighbors(gene)
                flag = self.__checkNeighbor(neighbors)
                if flag == 1:
                    self.clusterNum += 1
            else:
                pass
        self.clusterNum -= 1

    def localSearch(self):
        mutateGene = np.random.randint(self.chromosomeLen)
        # find the nodes connected to mutateGene
        neighbors = np.where(self.reduced_mtx[mutateGene]==1)[1]
        # find which cluster the mutateGene connects to the most
        mutateNum = stats.mode(self.cluster[neighbors])[0][0]
        fitNeighbor = np.where(self.cluster==mutateNum)[0]
        self.cluster[mutateGene] = mutateNum
        self.chromosome[mutateGene] = np.random.choice(fitNeighbor)

        self.clusterNum = len(np.unique(self.cluster))

    def __checkNeighbor(self, neighbors):
        flag = 0
        for gene in neighbors:
            belongCluster = self.cluster[gene]
            if belongCluster != 0:
                self.cluster[neighbors] = belongCluster
                return flag
        self.cluster[neighbors] = self.clusterNum
        flag = 1
        return flag

    def __getNeighbors(self, gene):
        neighbors = [gene]
        # print(self.chromosome)
        nxtGene = self.chromosome[gene]
        while True:
            if nxtGene not in neighbors:
                neighbors.append(nxtGene)
                nxtGene = self.chromosome[nxtGene]
            else:
                return neighbors

    def setModularity(self):
        Qvalue = 0
        for c in range(1, self.clusterNum+1):
            inValue = 0
            outValue = 0
            for i in range(self.chromosomeLen):
                for j in range(i + 1, self.chromosomeLen):
                    if self.cluster[i] == c and self.cluster[j] == c:
                        inValue += self.reduced_mtx[i, j]
                        # outValue += self.mtx[i, j]
                    elif self.cluster[i] != c and self.cluster[j] == c:
                        outValue += self.reduced_mtx[i, j]/2
                    elif self.cluster[j] != c and self.cluster[i] == c:
                        outValue += self.reduced_mtx[i, j]/2
            outValue += inValue
            Qvalue += inValue / self.edge
            Qvalue -= (outValue / self.edge)**2
        self.modularity = Qvalue
        # global nfe 
        # nfe += 1


class EGACD():
    def __init__(self, populationsSize, pc, generation, isParallel):
        self.populationSize = populationsSize
        self.chromosomeLen = reduced_mtx.shape[0]
        self.generation = generation
        self.pc = pc
        self.isParallel = isParallel
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
            # childrenA, childrenB = Chromosome(reduced_mtx), Chromosome(reduced_mtx)
            childrenA, childrenB = Chromosome(), Chromosome()
            # one point crossover operation
            childrenA.chromosome = np.concatenate((parentA.chromosome[:position+1], parentB.chromosome[position+1:]))
            childrenB.chromosome = np.concatenate((parentB.chromosome[:position+1], parentA.chromosome[position+1:]))
            return childrenA, childrenB
        else:
            return parentA, parentB

    def mutation(self, parent):
        children = parent
        # print(self.chromosomeLen)
        mutateGene = np.random.randint(self.chromosomeLen)
        neighbor = np.where(reduced_mtx[mutateGene]==1)[1]
        # print(neighbor)
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
        # print(population)
        for chromosome in population:
            chromosome.clusterize()
            chromosome.localSearch()
            chromosome.setModularity()
        return population
    
    def localSearch_p(self, chromosome):
        chromosome.clusterize()
        chromosome.localSearch()
        chromosome.setModularity()
        return chromosome

    def selection(self, population):
        selectedPopulation = sorted(population, key=lambda chromosome: chromosome.modularity, reverse=True)
        return selectedPopulation[:self.populationSize]

    def oneRun(self,population):
        populationDouble = self.operator(population)
        ## Parallel
        if self.isParallel:
            jobs_per_worker = int(np.ceil(len(populationDouble) / num_workers))
            # inputs = [populationDouble[i : i+jobs_per_worker] for i in range(0, len(populationDouble), jobs_per_worker)]
            pool_outputs = pool.map(self.localSearch_p, populationDouble)
            # print(pool_outputs)
            populationDouble = pool_outputs
            self.nfe += len(populationDouble)
            #print(len(populationDouble))
        else:
            populationDouble = self.localSearch(populationDouble)
        # populationDouble = self.MBCrossover(populationDouble)
        populationSelected = self.selection(populationDouble)
        bestModularity = populationSelected[0].modularity
        return bestModularity, populationSelected

    def doIt(self):
        population = self.initialization()
        self.nfe_mod_arr = []
        for _ in range(self.generation):
            bestModularity,population = self.oneRun(population)
            self.nfe_mod_arr.append([self.nfe,bestModularity])

        return bestModularity,population[0]

    def MBCrossover(self,population):
        linkageDictionary = self.linkageIdentify(population)
        BB = self.findBB(linkageDictionary)

        for i in range(len(population)):

            population[i].clusterize()
            population[i].setModularity()

            tmp = copy.deepcopy(population[i])
            for gene in range(len(BB)-1):
                tmp.chromosome[BB[gene]] = BB[gene+1]
            tmp.clusterize()
            tmp.setModularity()

            if tmp.modularity > population[i].modularity:
                population[i] = tmp
        return population


if __name__ == '__main__':
    path ='../soc-karate/soc-karate.mtx'
    mtx = loadDataset(path)
    isReduced = False
    isParallel = True
    cpu_count = os.cpu_count()
    num_workers = 5
    pool = Pool(num_workers)
    if (isReduced):
        obj = reducegraph(path, 0.2)
        index_selected, index_eliminated, mtx, reduced_mtx = \
            obj['index_selected'], obj['index_eliminated'], obj['original_mtx'], obj['reduced_mtx']
        edge = np.count_nonzero(mtx==1) / 2
        egacd = EGACD(30, 0.8, 50, isParallel)
    else:
        reduced_mtx = loadDataset(path)
        egacd = EGACD(30, 0.8, 50, isParallel)
    mod_arr  = []
    repeat = 5
    time_arr = []
    nfe_arr = []
    cluster_arr = []
    pro_arr = []
    for i in range(repeat):
        nfe = 0
        print("=== Start repeat [",i,"] ===")
        cpu_count = os.cpu_count()
        print('========Number of cpu: ' + str(cpu_count) + '===========')
        print('Use' + str(num_workers) + ' cores.')
        startTime = time.time()
        bestModularity, bestChromosome = egacd.doIt()
        if isReduced:
            concateReduced(bestChromosome)
            setModularity(bestChromosome, edge)
            bestModularity = bestChromosome.modularity
        print("Best Modularity: ",bestModularity)
        print("Best Chromosome: ",bestChromosome)
        print("Best cluster: ", bestChromosome.cluster)
        mod_arr.append(bestModularity)
        cluster_arr.append([int(i) for i in bestChromosome.cluster])
        pro_arr.append(egacd.nfe_mod_arr)

        time_arr.append(time.time()-startTime)
        print("Time: ",time_arr[i])

        nfe_arr.append(egacd.nfe)
        print("NFE: ",egacd.nfe)

    max_mod = max(mod_arr)
    max_index = mod_arr.index(max_mod)
    max_pro = np.array(pro_arr[max_index])
    
    print("BEST:", max(mod_arr))
    print("AVG:", sum(mod_arr)/repeat)
    print("AVG DURATION:",sum(time_arr)/repeat)
    print("AVG NFE:", sum(nfe_arr)/repeat)
    np.save('cluster.npy',{'cluster': cluster_arr[max_index], 'mod': max_mod})
    if not isReduced:
        writeProgress(max_pro)
