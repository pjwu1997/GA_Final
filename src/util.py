from scipy.io import mmread
import numpy as np
def loadDataset(path):
    mtx = mmread(path)
    mtx = mtx.todense()
    return mtx

def modularity(cluster, mtx):
    chromosomeLen = mtx.shape[0]
    edge = np.count_nonzero(mtx == 1) / 2
    clusterNum = np.unique(cluster).shape[0]

    Qvalue = 0
    for c in range(1, clusterNum+1):
        inValue = 0
        outValue = 0
        for i in range(chromosomeLen):
            for j in range(i + 1, chromosomeLen):
                if cluster[i] == c and cluster[j] == c:
                    inValue += mtx[i, j]
                elif cluster[i] != c and cluster[j] == c:
                    outValue += mtx[i, j]/2
                elif cluster[j] != c and cluster[i] == c:
                    outValue += mtx[i, j]/2
        outValue += inValue
        Qvalue += inValue / edge
        Qvalue -= (outValue / edge)**2
    return Qvalue

def transfer_cluster(cluster):
    cluster_np = np.zeros(34)
    clusterNum = 1
    for subcluster in cluster:
        for node in subcluster:
            cluster_np[node] = clusterNum
        clusterNum += 1
    return cluster_np