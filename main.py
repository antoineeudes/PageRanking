import numpy as np

def create_Adj(w):
    return np.ones((w,w)) - np.identity(w)

def google(Adj):
    w, h = Adj.shape

    # Choix de paramètres
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

if __name__=='__main__':
    Adj = create_Adj(3)
    print(google(Adj))
