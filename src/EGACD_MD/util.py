from scipy.io import mmread

def loadDataset(path):
    mtx = mmread(path)
    mtx = mtx.todense()
    return mtx