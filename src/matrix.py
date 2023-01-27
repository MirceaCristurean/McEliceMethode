from random import shuffle, randrange
import numpy as np
from time import time
timeout = time() + 2  # seconds


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


n = 30  # coloane
k = 20   # n-k randuris
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


H = remove_duplicates_rows(H)


def calcul_Lscor(H, S):
    LS, = S.nonzero()
    Lscor = []
    for index in range(0, n):
        nonzeros, = H[:, index].nonzero()
        intersect = np.intersect1d(nonzeros, LS)
        Lscor.append(len(intersect))
    return Lscor


# Reduced Row Echelon Form
def rref(A):
    m, n = A.shape
    i, j = 0, 0
    jb = []
    LCS = []  # permutarea
    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j
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
            # Subtract multiples of the pivot row from all the other rows
            for l in range(i+1, m):
                if A[l, j] == 1:
                    A[l, j:n] = np.logical_xor(A[l, j:n], A[i, j:n])
            i += 1
            j += 1
    for index in LCS:
        A = np.concatenate((A, np.vstack(A[:, index])), axis=1)
    for index in reversed(LCS):
        A = np.delete(A, index, 1)
    for j in range(m-1, 0, -1):
        L, = np.nonzero(A[0:j, j])
        for p in L:
            A[p, :] = np.logical_xor(A[j, :], A[p, :])
        np.identity(n-m)
        print(L)
    return A, jb


A, jb = rref(np.copy(H))
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
        break  # the loop after {timeout} amount of seconds
print("Finish")
print('e', e.nonzero())
print('y', y.nonzero())
print((H@y)*1)
print('Sindrome', sindrome*1)
print(np.equal(H@y, sindrome)*1)


# LS egal index in S fara 0         LS = index(1's in S)
# LS, = S.nonzero()
# print('LS', LS)
# Lscor = []
# for index in range(0, n):
#     nonzeros, = H[:, index].nonzero()
#     intersect = np.intersect1d(nonzeros, LS)
#     Lscor.append(len(intersect))

# # print('Lscor', Lscor)
# #   m = maxim( Lscor )
# M = max(Lscor)
# print('M', M)
# #   ListaMax = index ( m, Lscor )
# # for i in ListaMax: e[i] = e[i] xor 1
# ListMax, = (np.array(Lscor) >= M).nonzero()
# print('ListaMax', ListMax)
# # S = S xor H * e
# for columns in ListMax:
#     H[:][columns]

# Usefull
# [np.array].nonzero()
# np.logical_xor(arr1, arr2)
# np.intersect1d(arr1, arr2)

# Ecnrypt z = message * G + e
