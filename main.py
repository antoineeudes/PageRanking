import numpy as np
from numpy.linalg import norm

eps = 0.001

def create_Adj(w):
    Adj = np.ones((w,w)) - np.identity(w)
    Adj[-1,-2] = 0
    return Adj

def google(Adj):
    w, h = Adj.shape

    # Choix de parametres
    z = np.ones((1, h))/w
    alpha = 0.1
    e = np.ones((w,1))

    # Calcul de d
    sum_cols = Adj.sum(axis=1)
    d = (sum_cols==0).reshape((w,1))

    # Calcul de Pss
    degs = Adj.sum(axis=1)
    Pss = (Adj/degs)
    P = alpha*Pss + np.dot(d,z) + (1-alpha)*np.dot((e-d), z)
    Pprim = P.T
    return P, Pss, Pprim, d, z, alpha

def pi_iterative(P_prime):
    m, n = P_prime.shape
    pn = np.ones((n, 1))/n

    while True:
        p = pn
        pn = np.dot(P_prime, p)

        if norm(pn-p, np.inf) < eps:
            break

    return pn

def pi_iterative_sparse(Pss, d, z, alpha):
    Pss_prime = Pss.T
    m, n = Pss_prime.shape
    pn = np.ones((n, 1))/n
    e = np.ones((n, 1))

    M1 = alpha*Pss_prime
    M2 = np.transpose(np.dot(d+(1-alpha)*(e-d), z))

    while True:
        p = pn
        pn = np.dot(M1, p) + np.dot(M2, p)

        if norm(pn-p, np.inf) < eps:
            break

    return pn


def r(x):
    return x**2

def ergodique_markov(P):
    s = 0.
    pi = pi_iterative_sparse()
    for i in range(n):
        s += pi[i]*r(i)

    return s

def ergodique_markov_T(T, P):
    n, _ = P.shape

    P_pow = np.eye(n)
    s0 = 0
    for t in range(T):
        P_pow = np.dot(P_pow, P)
        s1 = 0
        for k in range(n):
            s2 = 0
            for i in range(n):
                s2 += P_pow[i, k]
            s1 += r(k)*s2
        s0 += s1

    return s0/T


if __name__=='__main__':
    Adj = create_Adj(10)
    print(google(Adj))
    P, Pss, Pprim, d, z, alpha = google(Adj)
    print(pi_iterative(Pprim))
    print(ergodique_markov_T(1000, P))
