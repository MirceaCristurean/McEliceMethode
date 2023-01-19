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
    #rand = randrange(0, length+1)
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


#Su = np.random.default_rng().choice(n, size=w, replace=False)
# vector de n zeros cu index de Su = 1
Su = np.zeros(n)
for index in np.random.default_rng().choice(n, size=w, replace=False):
    Su[index] = 1
Sv = np.zeros(n)
for index in np.random.default_rng().choice(n, size=w, replace=False):
    Sv[index] = 1
Pu = np.poly1d(Su)
Pv = np.poly1d(Sv)

oneX = np.poly1d([1, 0])  # 1x

#H = np.concatenate((PuSide, PvSide), axis=1)
identity = np.identity(n, dtype='bool')

H = matrixH(n-k, n, w)
# print_matrix(H)
print_rank(H)
print("Number of rows ", len(H))

H_final = np.copy(H)
delta = n-k-rank(H)
if delta != 0:
    print('Rank deficient by ', delta)
    H_temp = np.copy(H)
    while(delta > 0):
        for column in range(0, len(H_temp)):
            H_temp = np.delete(H_temp, column, 0)
            if(rank(H) == rank(H_temp)):
                H = np.copy(H_temp)
                H_final = np.copy(H_temp)
                break
            else:
                H_temp = np.copy(H)
        delta = delta - 1
else:
    print('Rank full')

H = H_final
# print_matrix(H_final)
print_rank(H_final)
lenH = len(H)
print("Number of rows ", lenH)


# alta metoda while rank < m-k, vertical join (H, shuffle 1*w 0*n-m)
def calcul_Lscor(H, S):
    LS, = S.nonzero()
    Lscor = []
    for index in range(0, n):
        nonzeros, = H[:, index].nonzero()
        intersect = np.intersect1d(nonzeros, LS)
        Lscor.append(len(intersect))
    return Lscor


#B = Matrix((H*1).astype(bool)).rref(pivots=False)
print(H*1)


# Reduced Row Echelon Form
def rref(A):
    m, n = A.shape
    #k = m - n
    i, j = 0, 0
    jb = []
    LCS = []  # permutarea

    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j
        k = np.argmax(np.abs(A[i:m, j])) + i
        print('k ', k)
        p = np.abs(A[k, j])
        print('p ', p*1)
        print('i ', i)
        if p*1 < 1:
            # The column is negligible, zero it out
            # A[i:m, j] = 0.0
            print('j', j)
            LCS.append(j)
            j = j + 1
        else:
            # Remember the column index
            jb.append(j)
            if i != k:
                # Swap the i-th and k-th rows
                A[[i, k], j:n] = A[[k, i], j:n]
                print("swap", i, k)
                print_matrix(A*1)
            # Divide the pivot row i by the pivot element A[i, j]
            # A[i, j:n] = A[i, j:n] / A[i, j]
            # Subtract multiples of the pivot row from all the other rows
            for l in range(i+1, m):
                if A[l, j] == 1:
                    A[l, j:n] = np.logical_xor(A[l, j:n], A[i, j:n])

            i += 1
            j += 1
            print('i', i, 'j', j)
            print()
    print('LCS', LCS)
    print_matrix(A*1)
    for index in LCS:
        A = np.concatenate((A, np.vstack(A[:, index])), axis=1)
    print('After Concat')
    print_matrix(A*1)
    for index in reversed(LCS):
        A = np.delete(A, index, 1)
    print('After Delete')

    for j in range(m-1, 0, -1):
        L, = np.nonzero(A[0:j, j])
        for p in L:
            A[p, :] = np.logical_xor(A[j, :], A[p, :])
        np.identity(n-m)
        print(L)

    # Finished
    print_matrix(A*1)
    return A, jb


A, jb = rref(np.copy(H))

# S un vector de eroare de lungime n-k
#S = randomVector(lenH)
e = randomVector(n, 3)
# print(H*1)
# print('e', e*1)

S = H @ e
sindrome = np.copy(S)
# print('S', S*1)

# y = np.zeros(n, dtype="bool")
# length, = S.nonzero()
# while len(length) != 0:
#     scor = calcul_Lscor(H, S)
#     print('scor', scor)
#     M = max(scor)
#     print('M', M)
#     ListMax, = (np.array(scor) >= M).nonzero()
#     print('ListaMax', ListMax)
#     V = np.zeros(n-k, dtype="bool")
#     x = np.zeros(n, dtype="bool")
#     for column in ListMax:
#         #print(np.transpose(H)[:][column] * 1)
#         V = V + np.transpose(H)[:][column]
#         x[column] = 1
#     #print('V', V)
#     S = np.logical_xor(S, V)
#     y = y + x

#     length, = S.nonzero()
#     if time() > timeout:
#         break  # the loop after {timeout} amount of seconds
# print('e', e.nonzero())
# print('y', y.nonzero())
# print((H@y)*1)
# print('Sindrome', sindrome*1)
# print(np.equal(H@y, sindrome)*1)


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
