import networkx as nx
import time
from numpy.lib.npyio import NpzFile
from src.ga_net import ga_community_detection
from src.util import loadDataset, modularity, transfer_cluster
import numpy as np


if __name__ == '__main__':
    path ='dataset/socfb-Caltech36/socfb-Caltech36.mtx'
    
    # adj is csr_matrix, mtx is coo_matrix
    # you can change their type via scipy library
    adj, mtx = loadDataset(path)
    modularity_bst = 0
    modularity_avg = 0
    nfe_bst = 0
    nfe_avg = 0
    start = time.time()
    for _ in range(1):
        cluster, nfe = ga_community_detection(adj, population=30, generation=100, r = 1.5)

        if nfe_bst == 0 or nfe < nfe_bst:
            nfe_bst = nfe
        nfe_avg += nfe

        cluster = transfer_cluster(cluster, mtx)

        Q = modularity(cluster, mtx)

        if Q > modularity_bst:
            modularity_bst = Q
        modularity_avg += Q
    end = time.time()
    print('mod: ' , modularity_bst)
    # print('mod_avg: ' , modularity_avg / 100)
    print('nfe: ' , nfe_bst)
    print('time: ', end - start)
    # print('nfe_avg: ' , nfe_avg / 100)



