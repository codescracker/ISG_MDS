from distance import *


def object_function(x, sign=1.0):
    n = len(Dt)
    m = len(Dt[0])
    UMatrix = U["U"]
    target = 0
    objective_weight = np.array([x[i] for i in range(0, m)])
    for i in range(0, n):
        for j in range(i + 1, n):
            target += L[i][j] * (
                (weightedL2(Dt[i], Dt[j], objective_weight)
                 - UMatrix[i][j] * weightedL2(Dt[i], Dt[j], Weight)
                 ) ** 2)
    return sign * target


def object_func_drive(x, sign=1.0):
    n = len(Dt)
    m = len(Dt[0])
    UMatrix = U["U"]
    targets = []
    objective_weight = np.array([x[i] for i in range(0, m)])
    for index in range(0, m):
        target = 0
        for i in range(0, n):
            for j in range(i + 1, n):
                part1 = L[i][j] * (
                    weightedL2(Dt[i], Dt[j], objective_weight)
                    - UMatrix[i][j] * weightedL2(Dt[i], Dt[j], Weight)
                )

                part2 = (Dt[i][index] - Dt[j][index]) ** 2

                target += sign * (part1 * part2)

        targets.append(2 * target)

    return np.array(targets)


def constrain(x):
    m = len(Dt[0])
    constrain = 0
    for i in range(0, m):
        constrain += x[i]
    constrain = constrain - 1
    return np.array([constrain])


def constrain_jac(x):
    m = len(Dt[0])
    driv = [1] * m
    return np.array(driv)


cons = ({
            'type': 'eq',
            'fun': constrain,
            'jac': constrain_jac
        },)

bnds = tuple([(0, None)] * len(Dt[0]))
