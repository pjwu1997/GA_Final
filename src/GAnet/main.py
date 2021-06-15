from os import chroot
from numpy.lib.function_base import _calculate_shapes
from scipy.io import mmread
import numpy as np
from src.util import concateReduced_ganet, loadDataset, modularity_ganet, reducegraph, concateReduced_ganet, transfer_cluster
import src.globals as globals
import scipy
import time
import src.globals
from src.GAnet.ga_net import ga_community_detection

if __name__ == '__main__':
    path ='./soc-karate/soc-karate.mtx'
    mtx = loadDataset(path)
    isReduced = True

    mod_arr  = []
    repeat = 2
    time_arr = []
    nfe_arr = []


    if (isReduced):
        obj = reducegraph(path, 0.2)
        globals.index_selected, globals.index_eliminated, globals.mtx, globals.reduced_mtx = \
            obj['index_selected'], obj['index_eliminated'], obj['original_mtx'], obj['reduced_mtx']
        edge = np.count_nonzero(mtx==1) / 2

        globals.adj = scipy.sparse.csr_matrix(globals.mtx)
        # globals.reduced_adj = scipy.sparse.csr_matrix(globals.reduced_mtx)
    else:
        globals.reduced_mtx = loadDataset(path)
    globals.reduced_adj = scipy.sparse.csr_matrix(globals.reduced_mtx)    

    for i in range(repeat):
        print("=== Start repeat [",i,"] ===")
        startTime = time.time()
        result, nfe =  ga_community_detection(globals.reduced_adj, 10, 30, r=1.5)
        cluster = transfer_cluster(result, globals.reduced_mtx)
        if isReduced:
            full_cluster, clusterNum = concateReduced_ganet(cluster)
            mod = modularity_ganet(full_cluster, globals.mtx)
        else:
            mod = modularity_ganet(cluster, globals.reduced_mtx)
        endTime = time.time()

        mod_arr.append(mod)
        nfe_arr.append(nfe)
        time_arr.append(endTime - startTime)

    print("BEST:", max(mod_arr))
    print("AVG:", sum(mod_arr)/repeat)
    print("AVG DURATION:",sum(time_arr)/repeat)
    print("AVG NFE:", sum(nfe_arr)/repeat)

    