import numpy as np
from scipy import stats
from src.util import loadDataset

class Chromosome():
    def __init__(self, mtx):
        # self.mtx = self.loadDataset(path)
        self.mtx = mtx
        self.edge = np.count_nonzero(self.mtx == 1) / 2
        self.chromosomeLen = self.mtx.shape[0]
        
        self.cluster = np.zeros(self.chromosomeLen, dtype='int')
        self.clusterNum = 1

        chromosome = np.zeros((self.chromosomeLen), dtype='int')
        for gene in range(self.chromosomeLen):
            neighbor = np.where(self.mtx[gene]==1)[1]
            chromosome[gene] = np.random.choice(neighbor)
        self.chromosome = chromosome
        self.modularity = None

    # def __repr__(self):
    #     return self.chromosome

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
        neighbors = np.where(self.mtx[mutateGene]==1)[1]

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
        nxtGene = self.chromosome[gene]
        while True:
            if nxtGene not in neighbors:
                neighbors.append(nxtGene)
                nxtGene = self.chromosome[nxtGene]
            else:
                return neighbors

    def setModularity(self):
        Qvalue = 0
        for c in range(1, self.clusterNum):
            inValue = 0
            outValue = 0
            for i in range(self.chromosomeLen):
                for j in range(i + 1, self.chromosomeLen):
                    if self.cluster[i] == c and self.cluster[j] == c:
                        inValue += self.mtx[i, j]
                        outValue += self.mtx[i, j]
                    elif self.cluster[i] != c and self.cluster[j] == c:
                        outValue += self.mtx[i, j]
                    elif self.cluster[j] != c and self.cluster[i] == c:
                        outValue +=self.mtx[i, j]
            Qvalue += inValue / self.edge
            Qvalue -= (outValue / (self.edge * 2))**2
        self.modularity = Qvalue
