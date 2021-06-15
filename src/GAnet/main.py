from os import chroot
import os
from numpy.lib.function_base import _calculate_shapes
from scipy.io import mmread
import numpy as np
from src.util import concateReduced_ganet, loadDataset, modularity_ganet, reducegraph, concateReduced_ganet, concateReduced_ganet_p, transfer_cluster
import src.globals as globals
import scipy
import time
import src.globals
from src.GAnet.ga_net import ga_community_detection
from multiprocessing import Pool

def run_ganet(graph, population, generation, r, reduced_mtx, mtx, isReduced, index_eliminated, index_selected):
    startTime = time.time()
    result, nfe = ga_community_detection(graph, population, generation, r)
    cluster = transfer_cluster(result, reduced_mtx)
    if isReduced:
        full_cluster, clusterNum = concateReduced_ganet_p(cluster, index_eliminated, index_selected, mtx)
        mod = modularity_ganet(full_cluster, mtx)
        cluster = full_cluster
    else:
        mod = modularity_ganet(cluster, reduced_mtx)
    endTime = time.time()
    duration = endTime - startTime
    obj = {'nfe': nfe, 'cluster': cluster, 'mod': mod, 'time': duration}
    print('nfe= ', nfe)
    print('cluster= ', cluster)
    print('mod= ', mod)
    print('time= ', duration)
    return {'nfe': nfe, 'cluster': cluster, 'mod': mod, 'time': duration}
    
if __name__ == '__main__':
    isReduced = False
    isParallel = True
    cpu_count = os.cpu_count()
    num_workers = 5
    pool = Pool(num_workers)

    path ='./soc-karate/soc-karate.mtx'
    mtx = loadDataset(path)

    mod_arr  = []
    repeat = 10
    time_arr = []
    nfe_arr = []


    if isReduced:
        obj = reducegraph(path, 0.2)
        globals.index_selected, globals.index_eliminated, globals.mtx, globals.reduced_mtx = \
            obj['index_selected'], obj['index_eliminated'], obj['original_mtx'], obj['reduced_mtx']
        edge = np.count_nonzero(mtx==1) / 2

        globals.adj = scipy.sparse.csr_matrix(globals.mtx)
        # globals.reduced_adj = scipy.sparse.csr_matrix(globals.reduced_mtx)
    else:
        globals.reduced_mtx = loadDataset(path)
    globals.reduced_adj = scipy.sparse.csr_matrix(globals.reduced_mtx)    
    parameters = [(globals.reduced_adj, 10, 30, 1.5, globals.reduced_mtx, globals.mtx, isReduced, globals.index_eliminated, globals.index_selected) for _ in range(repeat)]

    if isParallel:
        result = pool.starmap(run_ganet, parameters)
        print([a['mod'] for a in result])
        max_mod = max([a['mod'] for a in result])
        avg_mod = sum([a['mod'] for a in result]) / repeat
        max_index = [a['mod'] for a in result].index(max_mod)
        max_cluster = result[max_index]['cluster']
        avg_time = sum([a['time'] for a in result]) / repeat
        avg_nfe = sum([a['nfe'] for a in result]) / repeat

        print("BEST:", max_mod)
        print("AVG:", avg_mod)
        print("AVG DURATION:", avg_time)
        print("AVG NFE:", avg_nfe)
        print("Max cluster: ", max_cluster)
        max_cluster = [int(i) for i in max_cluster]
        np.save('graph.npy', {'cluster':max_cluster, 'mod': max_mod})

    else:
        for i in range(repeat):
            print("=== Start repeat [",i,"] ===")
            startTime = time.time()
            result, nfe =  ga_community_detection(globals.reduced_adj, 10, 30, r=1.5)
            cluster = transfer_cluster(result, globals.reduced_mtx)
            if isReduced:
                full_cluster, clusterNum = concateReduced_ganet(cluster)
                mod = modularity_ganet(full_cluster, globals.mtx)
                cluster = full_cluster
            else:
                mod = modularity_ganet(cluster, globals.reduced_mtx)
            endTime = time.time()

            mod_arr.append(mod)
            nfe_arr.append(nfe)
            time_arr.append(endTime - startTime)

            print("Best Modularity: ", mod)
            print("Best cluster: ", cluster)

        print("BEST:", max(mod_arr))
        print("AVG:", sum(mod_arr)/repeat)
        print("AVG DURATION:",sum(time_arr)/repeat)
        print("AVG NFE:", sum(nfe_arr)/repeat)

    