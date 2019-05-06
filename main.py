import numpy as np

def create_Adj(w, h):
    return np.one((w, h)) - np.eye((w, h))

def google(Adj):
    w, h = Adj.shape()
    P = np.zeros((w, h))
    Pss = np.zeros((w, h))
    Pprim = np.zeros((w, h))
    d = 0
    z = 0
    alpha = 0
    return P, Pss, Pprim, d, z, alpha

if __name__=='__main__':
    Adj = create_Adj((3,3))
    print(google(Adj))
