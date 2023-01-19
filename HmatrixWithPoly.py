from random import shuffle, randrange, random
import numpy as np
from time import time
from sympy import Matrix, GF, poly, gcdex, invert
from sympy.abc import x
from PIL import Image as im
timed = time()


def exportToPng(matrix):
    data = im.fromarray(np.array(matrix, dtype=np.uint8)*255)
    data.save('F{}.png'.format(random()))


def print_matrix(matrix, binary=True):
    if binary:
        matrix = matrix*1
        for row in matrix:
            print(row)
    else:
        for row in matrix:
            print(row)
    print()


def rref(A):

    m, n = A.shape
    i, j = 0, 0
    jb = []
    LCS = []

    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j
        k = np.argmax(np.abs(A[i:m, j])) + i
        p = np.abs(A[k, j])
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

            # Divide the pivot row i by the pivot element A[i, j]
            # A[i, j:n] = A[i, j:n] / A[i, j]
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
    for i in range(m-1, 0, -1):
        L, = np.nonzero(A[0:i, i])
        for j in L:
            A[j, :] = np.logical_xor(A[j, :], A[i, :])
    return A, jb, LCS


n = 10  # coloane
k = 8   # n-k randuri
w = 3  # numar de 1 pe linie


H_left = np.empty((n, n), dtype="int")
H_right = np.empty((n, n), dtype="int")
#Su = np.random.default_rng().choice(n, size=w, replace=False)
# vector de n zeros cu index de Su = 1
Su = np.zeros(n, dtype='int')
randArray = np.random.default_rng().choice(n, size=w, replace=False)
for index in randArray:
    Su[index] = 1
Pu = np.poly1d(Su)

print('Pu')
print(Pu)
F2 = GF(2)
f3 = poly(x, modulus=2)
f4 = poly(x, modulus=2)
for index in range(0, n):
    f3 = f3 + (Su[index] % 2)*x**index
    f4 = f4 + (Su[index] % 2)*x**index
print('F2 = ', f3-x)
pSu = f3-x
pSv = f4-x
pSui = invert(f3, x**n-1, domain=F2)
pSvi = invert(f3, x**n-1, domain=F2)


Sv = np.zeros(n, dtype='bool')
randArray = np.random.default_rng().choice(n, size=w, replace=False)
for index in randArray:
    Sv[index] = True
#Pv = np.poly1d(Sv)

for index in range(n):
    H_left[index] = np.roll(Su, index)
    H_right[index] = np.roll(Sv, index)

H = np.hstack((H_left, H_right))
print_matrix(H*1)
#H = Matrix.rref(H)
# exportToPng(H)
#input("Finished in {}".format(time()-timed))

# Usefull
# oneX = np.poly1d([1, 0])  # 1x
# H = np.concatenate((PuSide, PvSide), axis=1)
# identity = np.identity(n, dtype='bool')
# np.roll()
# [np.array].nonzero()
# np.logical_xor(arr1, arr2)
# np.intersect1d(arr1, arr2)

# Ecnrypt z = message * G + e
