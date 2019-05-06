import numpy as np
from numpy.linalg import norm

eps = 0.001

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

def pi_iterative(P_prime):
    m, n = P_prime.shape
    pn = np.ones(n)/n

    while True:
        p, pn = pn, np.dot(P_prime, p)

        if norm(pn-p, np.inf) < eps:
            break

    return p

if __name__=='__main__':
    Adj = create_Adj((3,3))
    print(google(Adj))
