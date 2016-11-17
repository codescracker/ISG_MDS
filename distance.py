import numpy as np


def weightedL2(a, b, w):
    q = a-b
    return np.sqrt((w*q*q).sum())


def dist_func(dataset, weight):
    length = len(dataset)
    res = np.array([([0]*length) for i in range(0, length)], dtype=np.float64)
    print("Current Weight:", weight)
    for i in range(0, length):
        for j in range(i, length):
            temp = weightedL2(dataset[i], dataset[j], weight)
            res[i][j] = temp
            res[j][i] = temp
    return res


def umatrix(dist_2_after, dist_2_before):
    further = 0
    closer = 0
    unchange = 0
    length = len(dist_2_before)
    u = np.array([[0] * length for i in range(0, length)], dtype=np.float64)
    for i in range(0, length):
        for j in range(i + 1, length):
            temp = dist_2_after[i][j] / dist_2_before[i][j]
            print("temp value: ", temp)
            if temp - 1.0 > 0.0001:
                further += 1
                print("further: ", further)
            elif 1.0 - temp > 0.0001:
                closer += 1
                print("closer: ", closer)
            else:
                unchange += 1
                print("unchange: ", unchange)
            u[i][j] = temp
            u[j][i] = temp
    return {"Further": further * 2, "Closer": closer * 2, "Unchange": unchange * 2, "U": u,
            "Total": length * (length - 1)}


def lmatrix(u):
    length = len(u["U"])
    unchange = u["Unchange"]
    closer = u["Closer"]
    further = u["Further"]
    u_matrix = u["U"]
    res = np.array([[0] * length for i in range(0, length)], dtype=np.float64)
    for i in range(0, length):
        for j in range(i + 1, length):
            if u_matrix[i][j] != 1:
                temp = unchange / (closer + further)
                res[i][j] = temp
                res[j][i] = temp
            else:
                res[i][j] = 1
                res[i][j] = 1
    return res


