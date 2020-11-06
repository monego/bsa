import numpy as np
import numpy.random as npr
import random as r


def bsa(f, N, D, maxcycle, mixrate, low, up):
    globalminimum = np.inf
    P, oldP = np.ones([N, D]), np.ones([N, D])
    u = [0 for i in range(D)]
    fitnessP = np.ones([N])
    fitnessT = np.ones([N])

    # Generate population

    for i in range(N):
        for j in range(D):
            P[i][j] = r.uniform(0, 1)*(up[j] - low[j]) + low[j]
            oldP[i][j] = r.uniform(0, 1)*(up[j] - low[j]) + low[j]
        fitnessP[i] = f(P[i])

    for i in range(maxcycle):

        # Selection I

        if r.uniform(0, 1) < r.uniform(0, 1):
            oldP = np.copy(P)

        npr.shuffle(oldP)

        # Generation of test population

        # # Mutação

        mutant = P + 3*npr.normal(0, 1)*(oldP - P)

        # # Crossover

        mapmtz = np.ones([N, D])

        if r.uniform(0, 1) < r.uniform(0, 1):
            for i in range(N):
                npr.shuffle(u)
                mapmtz[i][u[0:round(mixrate*r.uniform(0, 1)*D)]] = 0
        else:
            for i in range(N):
                mapmtz[i][r.randrange(0, D)] = 0

        # # Generation of trial population, T

        T = mutant

        for i in range(N):
            for j in range(D):
                if mapmtz[i][j] == 1:
                    T[i][j] = P[i][j]

        # # Boundary control mechanism

        for i in range(N):
            for j in range(D):
                if T[i][j] < low[j] or T[i][j] > up[j]:
                    T[i][j] = r.uniform(0, 1)*(up[j] - low[j]) + low[j]

        # Selection II

        for i in range(N):
            fitnessT[i] = f(T[i])

        for i in range(N):
            if fitnessT[i] < fitnessP[i]:
                fitnessP[i] = fitnessT[i]
                P[i] = T[i]

        bestFit = np.min(fitnessP)

        if bestFit < globalminimum:
            globalminimum = bestFit
            globalminimizer = P[np.argmin(fitnessP)]

    return globalminimum, globalminimizer
