from scipy.io import mmread
import numpy as np
import parallel.globals as globals
import copy
from scipy import stats

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

def concateReduced(Chromosome):
    """
    Construct full_chromosome and full_cluster.
    """
    full_chromosome = np.zeros(globals.mtx.shape[0])
    full_cluster = np.zeros(globals.mtx.shape[0])
    copy_chromosome = copy.deepcopy(Chromosome.chromosome)
    for index, neighbor in enumerate(copy_chromosome):
        copy_chromosome[index] = globals.index_selected[neighbor]
    full_chromosome[globals.index_selected] = copy_chromosome
    clusterNum = Chromosome.clusterNum
    # 看連哪個cluster最多
    # 選一個連
    # 轉成對應index, cluster
    full_cluster[globals.index_selected] = Chromosome.cluster
    # find the nodes connected to outsideGene
    for outsideGene in globals.index_eliminated:
        # print('outside: ',np.array(self.full_mtx[outsideGene])[0].shape)
        neighbors = np.where(np.array(globals.mtx[outsideGene])[0][globals.index_selected]==1)
        if not neighbors:
            full_cluster[outsideGene] = clusterNum + 1
            clusterNum += 1
        else:
            neighbors = neighbors[0]
        for index, neighbor in enumerate(neighbors):
            neighbors[index] = globals.index_selected[neighbor]
            # find which cluster the mutateGene connects to the most
            changeNum = stats.mode(full_cluster[neighbors])[0][0]
            fitNeighbor = np.where(full_cluster==changeNum)[0]
            full_cluster[outsideGene] = changeNum
            full_chromosome[outsideGene] = np.random.choice(fitNeighbor)
    Chromosome.chromosome, Chromosome.cluster, Chromosome.clusetNum = full_chromosome, full_cluster, clusterNum

def setModularity(Chromosome, edge):
    Qvalue = 0
    chromosomeLen = len(Chromosome.chromosome)
    for c in range(1, Chromosome.clusterNum+1):
        inValue = 0
        outValue = 0
        for i in range(chromosomeLen):
            for j in range(i + 1, chromosomeLen):
                if Chromosome.cluster[i] == c and Chromosome.cluster[j] == c:
                    inValue += globals.mtx[i, j]
                    # outValue += self.mtx[i, j]
                elif Chromosome.cluster[i] != c and Chromosome.cluster[j] == c:
                    outValue += globals.mtx[i, j]/2
                elif Chromosome.cluster[j] != c and Chromosome.cluster[i] == c:
                    outValue += globals.mtx[i, j]/2
        outValue += inValue
        Qvalue += inValue / edge
        Qvalue -= (outValue / edge)**2
    Chromosome.modularity = Qvalue