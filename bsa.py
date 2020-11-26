import random as r

from random import (uniform, gauss, randint)
from math import ceil
from numpy import (copy, arange, ones, zeros,
                   inf, argmin, sort, where, abs)
from numpy.random import shuffle
from scipy.optimize import minimize


def bsa_full(f, D, low, up, maxcycle, N, S, mixrate,
             Migration, MigrationSize, MigrationType,
             Epidemic, EpidemicRatio, EpidemicCounter,
             EpidemicTolerance, Endemic, EndemicRatio,
             Relaxation, MaxIterations):

    r.seed()

    # GA variables

    globalminimum = inf
    u = arange(0, D)
    fitnessP = ones([N])
    subP = zeros([S, N, D])
    subOldP = zeros([S, N, D])
    subFitnessP = zeros([S, N])
    subFitnessOldP = zeros([S, N])
    globalminimizer = zeros(D)
    subFitnessP_sorted = zeros([S, N])
    ctr = zeros([S])
    bestFit = inf
    bestPop = 0
    bestCrom = 0
    subFitnessT = ones([S, N])

    # Default parameters

    MigrationSize = int(N*MigrationSize)
    MortalityRatio = copy(EpidemicRatio)
    EpidemicRatio = int(N*EpidemicRatio)
    EndemicRatio = int(N*EndemicRatio)

    # Support variables

    RNDMIG1, RNDMIG2 = arange(S, dtype='uint8'), arange(S, dtype='uint8')

    l_bfgs_b_bounds = tuple(zip(low, up))

    bestPopFitness = []
    bestPopIndex = 0

    # Generate population

    for k in range(S):
        for i in range(N):
            for j in range(D):
                subP[k][i][j] = uniform(0, 1)*(up[j] - low[j]) + low[j]
                subOldP[k][i][j] = uniform(0, 1)*(up[j] - low[j]) + low[j]
            fitnessP[i] = f(subP[k][i])
        subFitnessP[k] = fitnessP

    for maxc in range(maxcycle):
        for k in range(S):

            # Selection I

            if uniform(0, 1) < uniform(0, 1):
                subOldP[k] = copy(subP[k])

            shuffle(subOldP[k])

            # Test population

            # Mutation

            mutant = subP[k] + 3*gauss(0, 1)*(subOldP[k] - subP[k])

            # Crossover

            mapmtz = ones([N, D])

            if uniform(0, 1) < uniform(0, 1):
                for i in range(N):
                    shuffle(u)
                    mapmtz[i][u[0:int(ceil(mixrate*uniform(0, 1)*D))]] = 0
            else:
                for i in range(N):
                    mapmtz[i][randint(0, D-1)] = 0

            # Generation of trial population, T

            T = mutant

            for i in range(N):
                for j in range(D):
                    if mapmtz[i][j] == 1:
                        T[i][j] = subP[k][i][j]

            # Boundary control mechanism

            for i in range(N):
                for j in range(D):
                    if T[i][j] < low[j] or T[i][j] > up[j]:
                        T[i][j] = uniform(0, 1)*(up[j] - low[j]) + low[j]

            # Selection II

            for i in range(N):
                subFitnessT[k][i] = f(T[i])

            for i in range(N):
                if subFitnessT[k][i] < subFitnessP[k][i] and subFitnessT[k][i] >= 0:
                    subFitnessP[k][i] = subFitnessT[k][i]
                    subP[k][i] = T[i]

        # Migration

        if Migration is True:

            for k in range(S):
                    subFitnessP_sorted[k] = sort(subFitnessP[k])

            subFitnessP_backup = copy(subFitnessP)

            if MigrationType == "ring":

                for k in range(-1, S-1):
                    for m in range(MigrationSize):
                        randindex = randint(0, N-1)
                        ind_best = where(subFitnessP[k]==subFitnessP_sorted[k][m])[0][0]
                        if subFitnessP[k][ind_best] < subFitnessP[k+1][randindex]:
                            subOldP[k+1][randindex] = subP[k+1][randindex]
                            subP[k+1][randindex] = subP[k][ind_best]
                            subFitnessOldP[k+1][randindex] = subFitnessP_backup[k+1][randindex]
                            subFitnessP_backup[k+1][randindex] = subFitnessP_backup[k][ind_best]

                subFitnessP = subFitnessP_backup

            elif MigrationType == "random":
                shuffle(RNDMIG1)
                shuffle(RNDMIG2)

                for pfrom, pto in zip(RNDMIG1, RNDMIG2):
                    if pfrom != pto:
                        for m in range(MigrationSize):
                            randindex = randint(0, N-1)
                            ind_best = where(subFitnessP[pfrom] == subFitnessP_sorted[pfrom][m])[0][0]
                            if subFitnessP[pfrom][ind_best] < subFitnessP[pto][randindex]:
                                subOldP[pfrom][randindex] = subP[pto][randindex]
                                subP[pto][randindex] = subP[pfrom][ind_best]
                                subFitnessOldP[pto][randindex] = subFitnessP_backup[pto][randindex]
                                subFitnessP_backup[pto][randindex] = subFitnessP_backup[pfrom][ind_best]

                subFitnessP = subFitnessP_backup

            else:
                raise TypeError("Error: Invalid migration type")

        # Epidemic

        if Epidemic is True:
            for k in range(S):
                #subFitnessP_sorted[k] = sort(subFitnessP[k])
                #ind_best = where(subFitnessP[k]==subFitnessP_sorted[k][0])[0][0]
                ind_best = argmin(subFitnessP[k])
                ind_best_old = argmin(subFitnessOldP[k])
                diffF = subFitnessOldP[k][ind_best_old] - subFitnessP[k][ind_best]
                if diffF < 0:
                    diffF = -diffF
                if diffF < EpidemicTolerance:
                    ctr[k] += 1
                if ctr[k] == EpidemicCounter:
                    for i in range(EpidemicRatio):
                        if uniform(0, 1) < MortalityRatio:
                            subOldP[k][i] = subP[k][i]
                            subFitnessOldP[k][i] = subFitnessP[k][i]
                            subP[k][i] = uniform(0, 1)*(up[j] - low[j]) + low[j]
                            subFitnessP[k][i] = f(subP[k][i])
                    ctr[k] = 0

        # Endemic

        if Endemic is True:
            for k in range(S):
                for e in range(EndemicRatio):
                    randindex = randint(0, N-1)
                    subOldP[k][randindex] = subP[k][randindex]
                    subFitnessOldP[k][randindex] = subFitnessP[k][randindex]
                    subP[k][randindex] = uniform(0, 1)*(up[j] - low[j]) + low[j]
                    subFitnessP[k][randindex] = f(subP[k][randindex])

        # Local relaxation

        if Relaxation is True:
            for k in range(S):
                for i in range(N):
                    pop_tmp = minimize(f, subP[k][i], method='L-BFGS-B', tol=1e-1, bounds=l_bfgs_b_bounds, options={'gtol' : 0.9, 'maxiter' : MaxIterations, 'maxfun' : MaxIterations}).get('x')
                    fitness_tmp = f(pop_tmp)
                    if (fitness_tmp < subFitnessP[k][i]):
                        subOldP[k][i] = subP[k][i]
                        subP[k][i] = pop_tmp
                        subFitnessOldP[k][i] = subFitnessP[k][i]
                        subFitnessP[k][i] = fitness_tmp

        # Calculate best population

        bestPopIndex = 0

        for k in range(S-1):
            if min(abs(subFitnessP[k+1])) < min(abs(subFitnessP[k])):
                bestPopIndex = k+1

        bestPopFitness.append(min(subFitnessP[bestPopIndex]))

    for k in range(S):
        if min(abs(subFitnessP[k])) < bestFit:
            bestFit = min(abs(subFitnessP[k]))
            bestPop = k
            bestCrom = where(abs(subFitnessP[k]) == bestFit)[0][0]

    if bestFit < globalminimum:
        globalminimum = bestFit
        globalminimizer = subP[bestPop][bestCrom]

    return globalminimum, globalminimizer
