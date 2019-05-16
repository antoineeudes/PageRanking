import numpy as np
from numpy.linalg import norm

eps = 0.001


def create_Adj(w):
    Adj = np.ones((w,w)) - np.identity(w)
    Adj[-1,3] = 0
    Adj[-1,-2] = 0
    Adj[-5,-6] = 0
    Adj[-2,-3] = 0
    Adj[-2,-1] = 0
    Adj[-2,0] = 0
    Adj[1,0] = 0
    Adj[5:, 0] = 0
    Adj[5:, 1] = 0

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

Adj = create_Adj(10)
P, Pss, Pprim, d, z, alpha = google(Adj)


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

def binary_encoding(integer, digits):
    encoded = bin(integer)[2:]
    while len(encoded)<digits:
        encoded = '0' + encoded
    return encoded

def create_candidate_matrix(binary_integer):
    return np.array([int(s) for s in binary_integer])

def get_candidate_matrices(nb_rows, nb_columns):
    nb_candidates = 2**(nb_rows*nb_columns)
    candidate_matrices = np.zeros((nb_candidates, nb_rows, nb_columns))
    digits = nb_rows*nb_columns
    for i in range(nb_candidates):
        candidate_matrices[i, :, :] = create_candidate_matrix(binary_encoding(i, digits)).reshape((nb_rows, nb_columns))
    return candidate_matrices

def optimizePageRank(n, p=2, m=None):
    if m==None:
        m = n//2
    Adj = create_Adj(n)
    Adj[0:p, m+1:] = 0
    candidate_matrices = get_candidate_matrices(p, n-1-m)
    optimal_matrice = np.zeros((p, n-m))
    optimal_score = -float('inf')
    for matrice in candidate_matrices:
        Adj[0:2, m+1:] = matrice
        P, Pss, Pprim, d, z, alpha = google(Adj)
        pi = pi_iterative(P.T)
        score = np.sum([pi[i] for i in range(m)])
        if optimal_score<score:
            optimal_score = score
            optimal_matrice = matrice
    Adj[0:2, m+1:] = optimal_matrice
    return Adj

def r(x):
    return x**2

r_vect = np.vectorize(r)

def pageRank(P):
    eigen, M = np.linalg.eig(P.T)
    n, _ = P.shape
    return M.T[0]/np.sum(M.T[0])

def ergodique_markov(P):
    s = 0.
    pi = pi_iterative_sparse(Pss, d, z, alpha)
    n, _ = P.shape
    for i in range(n):
        s += pi[i]*r(i)
    return s

def trajectory(P, T):
    n, _ = P.shape
    X = np.zeros(T)
    X.astype(int)

    X[0] = np.random.randint(n)
    for t in range(1, T):
        dist = P[int(X[t-1]), :] # Probas de se déplacer dans un autre état
        u = np.random.rand()
        dist_cum = np.cumsum(dist)
        for k in range(len(dist)): # Simule un déplacement
            if u < dist_cum[k]:
                X[t] = k
                break
    return X


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
                s2 += P_pow[k, i]
            s1 += r(k)*s2
        s0 += s1

    return s0/T

def ergodique_markov_T_monte_carlo(T, P, N):
    means = []
    for k in range(N):
        if k % (N/10) == 0:
            print('Trajectory {}'.format(k))
        X = trajectory(P, T) # Simulate a trajectory
        means.append(np.mean(r_vect(X)))

    return np.mean(means)


def solve_linear_system(P):
    n, _ = P.shape
    A = P[:-1, :-1]-np.identity(n-1)
    b = -P[:-1,-1]
    return np.linalg.solve(A, b)



if __name__=='__main__':
    # print(google(Adj))
    # print(pi_iterative(Pprim))
    # print(ergodique_markov_T(1000, P))
    # print(ergodique_markov(P))
    # print(solve_linear_system(P))
    # print(trajectory(P, 100))
    # print(ergodique_markov_T(100, P, 1000))
    # print(optimizePageRank(10))
    print(pageRank(P))
    # print(optimizePageRank(10))
