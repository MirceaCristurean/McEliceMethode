from random import shuffle, randrange
import numpy as np
from time import time
import matplotlib.pyplot as plt


def randomMatrix(rows, columns):
    # matrice binara random
    matrix = np.zeros((rows, columns), dtype='bool')
    for index in range(0, rows):
        nr = randrange(0, columns+1)
        new_row = [True]*nr+[False]*(columns-nr)
        shuffle(new_row)
        matrix[index] = list(new_row)
    return matrix


def randomVector(length, w):
    # rand = randrange(0, length+1)
    arr = [True]*w+[False]*(length-w)
    shuffle(arr)
    return np.array(arr, dtype='bool')


def matrixH(rows, columns, w):
    # initializare matrice cu zero binar
    matrix = np.zeros((rows, columns), dtype='bool')
    for index in range(0, rows):
        new_row = [True]*w+[False]*(columns-w)
        shuffle(new_row)
        matrix[index] = list(new_row)
    return matrix


def rank(m):
    return np.linalg.matrix_rank(m)


def print_matrix(matrix, binary=True):
    if binary:
        matrix = matrix*1
        for row in matrix:
            print(row)
    else:
        for row in matrix:
            print(row)
    print()


def print_rank(matrix):
    print("The rank is ", np.linalg.matrix_rank(matrix))


def calcul_Lscor(H, S):
    LS, = S.nonzero()
    Lscor = []
    for index in range(0, n):
        nonzeros, = H[:, index].nonzero()
        intersect = np.intersect1d(nonzeros, LS)
        Lscor.append(len(intersect))
    return Lscor


def rref(A):
    m, n = A.shape
    i, j = 0, 0
    jb = []
    LCS = []
    while i < m and j < n:
        k = np.argmax(np.abs(A[i:m, j])) + i
        p = np.abs(A[k, j])
        if p*1 < 1:
            # The column is negligible, zero it out
            LCS.append(j)
            j = j + 1
        else:
            # Remember the column index
            jb.append(j)
            if i != k:
                # Swap the i-th and k-th rows
                A[[i, k], j:n] = A[[k, i], j:n]
            for l in range(i+1, m):
                if A[l, j] == 1:
                    A[l, j:n] = np.logical_xor(A[l, j:n], A[i, j:n])
            i += 1
            j += 1
    for index in LCS:
        A = np.concatenate((A, np.vstack(A[:, index])), axis=1)
    for index in reversed(LCS):
        A = np.delete(A, index, 1)
    for i in range(m-1, 0, -1):
        L, = np.nonzero(A[0:i, i])
        for j in L:
            A[j, :] = np.logical_xor(A[j, :], A[i, :])
    return A, jb, LCS


time_spent = time()
timeout = time() + 2  # seconds for timeout
n = 20  # coloane
k = 10   # n-k randuri
w = 5  # numar de 1 pe linie

H = matrixH(n-k, n, w)


def remove_duplicates_rows(H):
    H_final = np.copy(H)
    delta = n-k-rank(H)
    if delta != 0:
        H_temp = np.copy(H)
        while (delta > 0):
            for column in range(0, len(H_temp)):
                H_temp = np.delete(H_temp, column, 0)
                if (rank(H) == rank(H_temp)):
                    H = np.copy(H_temp)
                    H_final = np.copy(H_temp)
                    break
                else:
                    H_temp = np.copy(H)
            delta = delta - 1
    else:
        print('Rank full')
    return H_final

# H = remove_duplicates_rows(H)


def encrypt(H):
    Hsys, jb, LCS = rref(np.copy(H))
    k, n = H.shape
    Hperm = np.copy(H)
    for index in reversed(LCS):
        Hperm = np.delete(Hperm, index, 1)
    for index in LCS:
        Hperm = np.concatenate((Hperm, np.vstack(Hperm[:, index])), axis=1)
    #G = np.zeros((H.shape), dtype="bool")
    G = np.concatenate(
        (np.transpose(Hsys[:, n-k:n]), np.identity(k, dtype="bool")), axis=1)
    x = np.dot(G*(-1), np.transpose(Hsys*1))
    e = randomVector(n, 3)
    S = H @ e
    sindrome = np.copy(S)
    y = np.zeros(n, dtype="bool")
    length, = S.nonzero()
    while len(length) != 0:
        scor = calcul_Lscor(H, S)
        M = max(scor)
        ListMax, = (np.array(scor) >= M).nonzero()
        V = np.zeros(n-k, dtype="bool")
        x = np.zeros(n, dtype="bool")
        for column in ListMax:
            V = V + np.transpose(H)[:][column]
            x[column] = 1
        S = np.logical_xor(S, V)
        y = y + x
        length, = S.nonzero()
        if time() > timeout:
            print("failed")
        break  # the loop after {timeout} amount of seconds
    print(f"Time spent: {time()-time_spent} seconds")
    return y


#encrypted_message = encrypt(H)

# def decrypt(message)
w = 5
n_values = [100, 150, 200, 500, 800, 1000, 2000, 3000, 4000, 5000, 10000]
times = []

for n in n_values:
    start = time()
    H = matrixH(round(n/2), n, w)
    encrypt(H)
    end = time()
    times.append(end - start)
    print("n:", n, "time:", end - start)

plt.plot(n_values, times)
plt.xlabel('n values')
plt.ylabel('Running time (seconds)')
plt.title('Running time')
plt.show()